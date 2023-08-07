package main

import (
	"fmt"
	"time"

	"github.com/pkoukk/tiktoken-go"
)

func tokensPlay() {
	encoding := must(tiktoken.GetEncoding("r50k_base"))
	tokens := encoding.Encode("Now is the time for all good men to come to the aid of their party.", nil, nil)
	fmt.Printf("Encode: %v\n", tokens)
	fmt.Printf("Decode: %s\n\n", encoding.Decode(tokens))
	printTokensInColor(encoding, "Now is the time for all good men to come to the aid of their party.")

	time.Sleep(time.Second * 5)
	for i, allTokens := 0, getAllTokens(encoding); i < len(allTokens); i++ {
		fmt.Printf("%d: %q\n", i, allTokens[i])
	}
}

func getAllTokens(encoding *tiktoken.Tiktoken) []string {
	tokens := []string{}
	for i := 0; true; i++ {
		str := encoding.Decode([]int{i})
		if str == "" {
			break
		}
		tokens = append(tokens, str)
		//fmt.Printf("%d: %s\n", i, str) // Testing
	}
	return tokens
}

func printTokensInColor(encoding *tiktoken.Tiktoken, s string) {
	colors := []Color{BackgroundMagenta, BackgroundGreen, BackgroundCyan, BackgroundRed, BackgroundBlue}
	for tokens, i := encoding.Encode(s, nil, nil), 0; i < len(tokens); i++ {
		fmt.Printf("%s%s", colors[i%len(colors)], encoding.Decode(tokens[i:i+1]))
	}
	fmt.Println(ResetAllAttributes)
}

type Color string

const ( // https://learn.microsoft.com/en-us/windows/console/console-virtual-terminal-sequences
	ResetAllAttributes Color = "\x1b[0m"
	Negative           Color = "\x1b[7m"  // Swaps foreground and background colors
	Positive           Color = "\x1b[27m" // Returns foreground/background to normal

	// Foregound attributes
	ForewardBold            Color = "\x1b[1m"
	ForewardNoBold          Color = "\x1b[22m"
	ForewardUnderline       Color = "\x1b[4m"
	ForewardNoUnderline     Color = "\x1b[24m"
	ForewardBlack           Color = "\x1b[30m"
	ForewardRed             Color = "\x1b[31m"
	ForewardGreen           Color = "\x1b[32m"
	ForewardYellow          Color = "\x1b[33m"
	ForewardBlue            Color = "\x1b[34m"
	ForewardMagenta         Color = "\x1b[35m"
	ForewardCyan            Color = "\x1b[36m"
	ForewardWhite           Color = "\x1b[37m"
	ForewardExtended        Color = "\x1b[38m" // Applies extended color value to the foreground
	ForewardDefault         Color = "\x1b[39m" // Applies only the foreground portion of the defaults (see 0)
	BrightForegroundBlack   Color = "\x1b[90m"
	BrightForegroundRed     Color = "\x1b[91m"
	BrightForegroundGreen   Color = "\x1b[92m"
	BrightForegroundYellow  Color = "\x1b[93m"
	BrightForegroundBlue    Color = "\x1b[94m"
	BrightForegroundMagenta Color = "\x1b[95m"
	BrightForegroundCyan    Color = "\x1b[96m"
	BrightForegroundWhite   Color = "\x1b[97m"

	// Background attributes
	BackgroundBlack         Color = "\x1b[40m"
	BackgroundRed           Color = "\x1b[41m"
	BackgroundGreen         Color = "\x1b[42m"
	BackgroundYellow        Color = "\x1b[43m"
	BackgroundBlue          Color = "\x1b[44m"
	BackgroundMagenta       Color = "\x1b[45m"
	BackgroundCyan          Color = "\x1b[46m"
	BackgroundWhite         Color = "\x1b[47m"
	BackgroundExtended      Color = "\x1b[48m" // Applies extended color value to the background
	BackgroundDefault       Color = "\x1b[49m" // Applies only the background portion of the defaults (see 0)
	BrightBackgroundBlack   Color = "\x1b[100m"
	BrightBackgroundRed     Color = "\x1b[101m"
	BrightBackgroundGreen   Color = "\x1b[102m"
	BrightBackgroundYellow  Color = "\x1b[103m"
	BrightBackgroundBlue    Color = "\x1b[104m"
	BrightBackgroundMagenta Color = "\x1b[105m"
	BrightBackgroundCyan    Color = "\x1b[106m"
	BrightBackgroundWhite   Color = "\x1b[107m"
)
