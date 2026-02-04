package diag

import (
	"encoding/json"
	"fmt"
	"log"
)

func LogJSON(enabled bool, fn func(string, string), label string, value any) {
	data, err := json.Marshal(value)
	if err != nil {
		if fn != nil {
			fn(label, fmt.Sprintf("<marshal error: %v>", err))
			return
		}
		if enabled {
			log.Printf("%s: <marshal error: %v>", label, err)
		}
		return
	}
	if fn != nil {
		fn(label, string(data))
		return
	}
	if !enabled {
		return
	}
	log.Printf("%s: %s", label, string(data))
}

func LogText(enabled bool, fn func(string, string), label string, text string) {
	if fn != nil {
		fn(label, text)
		return
	}
	if !enabled {
		return
	}
	log.Printf("%s: %s", label, text)
}
