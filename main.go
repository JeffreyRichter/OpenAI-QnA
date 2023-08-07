package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/to"
	"github.com/Azure/azure-sdk-for-go/sdk/cognitiveservices/azopenai"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Expected 'createdb' or 'chat' subcommands")
		os.Exit(1)
	}

	switch os.Args[1] { // Check which subcommand is invoked.
	case "createdb":
		createDB(os.Args[2:])
	case "chat":
		chat(os.Args[2:])
	default:
		fmt.Println("Expected 'createdb' or 'chat' subcommands")
		os.Exit(1)
	}
}

func streaming() {
	// https://ms.portal.azure.com/#@microsoft.onmicrosoft.com/resource/subscriptions/faa080af-c1d8-40ad-9cce-e1a450ca5b57/resourceGroups/openai-shared/providers/Microsoft.CognitiveServices/accounts/openai-shared/cskeys
	kc, _ := azopenai.NewKeyCredential("")
	client := must(azopenai.NewClientWithKeyCredential("https://openai-shared.openai.azure.com/", kc, "ChatGPT", nil))
	resp := must(client.GetCompletionsStream(context.TODO(), azopenai.CompletionsOptions{
		Prompt:      []string{"What is Azure OpenAI?"},
		MaxTokens:   to.Ptr(int32(2048)),
		Temperature: to.Ptr(float32(0.0)),
	}, nil))

	for {
		entry, err := resp.CompletionsStream.Read()
		if errors.Is(err, io.EOF) {
			fmt.Printf("\n *** NO MORE COMPLETIONS ***")
			break
		}
		must(entry, err)

		for _, choice := range entry.Choices {
			fmt.Printf("%s", *choice.Text)
		}
	}
}

func must[R any](r R, err error) R {
	if err != nil {
		panic(err)
	}
	return r
}

type Cart struct {
	Items []LineItem
}

type LineItem struct {
	Product  Product
	Quantity int
}

type Product string

const (
	ProductBakery         Product = "Bakery"
	ProductLatteDrinks    Product = "Latte"
	ProductEspressoDrinks Product = "Espresso"
	ProductCoffeeDrinks   Product = "Coffee"
)

type BakeryProducts struct {
	Name    BakeryProductsName
	Options []BakeryProductsOption
}

type BakeryProductsName string

const (
	BakeryProductsNameAppleBranMuffin BakeryProductsName = "Apple Bran Muffin"
	BakeryProductsNameBlueberryMuffin BakeryProductsName = "Blueberry Muffin"
	BakeryProductsNameBagel           BakeryProductsName = "Bagel"
)

type BakeryProductsOption struct {
}
