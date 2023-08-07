// https://abdullin.com/llm/how-to-segment-text-for-embeddings/
// https://www.pinecone.io/learn/chunking-strategies/
package main

import (
	"bufio"
	"context"
	"encoding/gob"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/Azure/azure-sdk-for-go/sdk/cognitiveservices/azopenai"
	"github.com/ledongthuc/pdf"
	"golang.org/x/exp/slices"
)

func createDB(arguments []string) {
	cmd := flag.NewFlagSet("createdb", flag.ExitOnError)
	params := createDBCmdParams{}
	cmd.StringVar(&params.srcPath, "src", "", "path to the source file used to create the vector DB")
	cmd.StringVar(&params.dbPathname, "db", "", "path to the vector DB output file")
	cmd.StringVar(&params.clientUrl, "url", "", "URL of the OpenAI service for embeddings")
	cmd.StringVar(&params.clientAPIKey, "apikey", "", "API key used to authenticate against the OpenAI service")
	cmd.Parse(arguments)

	kc, _ := azopenai.NewKeyCredential(params.clientAPIKey)
	embedClient := must(azopenai.NewClientWithKeyCredential(params.clientUrl, kc, "text-embedding-ada-002-2", nil))

	pdf.DebugOn = true
	pdfFile, pdfReader, err := pdf.Open(params.srcPath)
	must(0, err)
	defer pdfFile.Close()
	r := must(pdfReader.GetPlainText())
	chunks := textSplitter(r, textSplitterOptions{chunkSize: 500, chunkOverlap: 100})
	entries := []Entry{}
	for _, chunk := range chunks {
		// TODO: This can be made more efficient by sending mutiple chunks in a single call up to max-token; see https://platform.openai.com/docs/api-reference/embeddings
		response := must(embedClient.GetEmbeddings(context.TODO(), azopenai.EmbeddingsOptions{Input: []string{chunk}}, nil))
		fmt.Println(chunk)
		entries = append(entries, Entry{ID: ID(chunk), Metadata: nil, Vector: response.Embeddings.Data[0].Embedding})
	}
	slices.SortFunc(entries, func(i, j Entry) bool { return i.ID < j.ID }) // Sort the entries by ID

	// Save the vectors to the DB file
	f := must(os.Create(params.dbPathname))
	defer f.Close()
	must(0, gob.NewEncoder(f).Encode(entries))
}

type createDBCmdParams struct {
	srcPath      string
	dbPathname   string
	clientUrl    string // "https://openai-shared.openai.azure.com/"
	clientAPIKey string
}

type textSplitterOptions struct {
	//sep          []string
	chunkSize    int // Default=4000
	chunkOverlap int // Default=200
}

func textSplitter(text io.Reader, o textSplitterOptions) []string {
	// https://js.langchain.com/docs/modules/indexes/text_splitters/examples/recursive_character
	// https://github.com/hwchase17/langchain/blob/master/langchain/text_splitter.py#L56

	// Split text into slice of words
	words := []string{}
	scanner := bufio.NewScanner(text)
	scanner.Split(bufio.ScanWords)
	for scanner.Scan() {
		words = append(words, scanner.Text())
	}
	must(0, scanner.Err())

	chunks := []string{}
	for index := 0; index < len(words[index:]); {
		numWordsInChunk := len(words[index:]) // Grab at most chunkSize words into a chunk
		if numWordsInChunk > o.chunkSize {
			numWordsInChunk = o.chunkSize
		}
		chunks = append(chunks, strings.Join(words[index:index+numWordsInChunk], " "))
		index += numWordsInChunk - o.chunkOverlap
	}
	return chunks
}

//with "\n\n", then "\n", then " ". This
