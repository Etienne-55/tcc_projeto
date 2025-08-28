package repositories

import (
	"database/sql"
	"net/http"
	"fmt"
	"os"
	"io"
	"encoding/json"
	"bytes"
	"golang_crud/models"
	"log"
	"strconv"
	"strings"
)

func getOllamaURL() string {
    ollamaURL := os.Getenv("OLLAMA_URL")
    if ollamaURL != "" {
        return ollamaURL
    }
    // Detect Docker environment
    if _, exists := os.LookupEnv("DOCKER_CONTAINER"); exists {
        return "http://host.docker.internal:11434"
    }
    return "http://localhost:11434"
}

// CreateDocument fetches embedding from Ollama and saves to Postgres
func CreateDocument(db *sql.DB, doc *models.Document) error {
    // Prepare Ollama request
    ollamaReq := map[string]string{
        "model":  "nomic-embed-text",
        "prompt": doc.Content,
    }
    requestBody, err := json.Marshal(ollamaReq)
    if err != nil {
        log.Printf("Failed to marshal Ollama request: %v", err)
        return fmt.Errorf("failed to prepare embedding request: %v", err)
    }

    // Call Ollama
    endpoint := getOllamaURL() + "/api/embeddings"
    log.Printf("Sending request to Ollama: %s, URL: %s", string(requestBody), endpoint)
    resp, err := http.Post(endpoint, "application/json", bytes.NewBuffer(requestBody))
    if err != nil {
        log.Printf("Ollama connection error: %v", err)
        return fmt.Errorf("failed to connect to embedding service: %v", err)
    }
    defer resp.Body.Close()

    body, err := io.ReadAll(resp.Body)
    if err != nil {
        log.Printf("Error reading Ollama response: %v", err)
        return fmt.Errorf("failed to read embedding response: %v", err)
    }
    log.Printf("Ollama response: status=%d, body=%s", resp.StatusCode, string(body))

    if resp.StatusCode != http.StatusOK {
        log.Printf("Ollama returned non-200 status: %d", resp.StatusCode)
        return fmt.Errorf("embedding service returned status: %d", resp.StatusCode)
    }

    // Parse Ollama response
    var ollamaResp struct {
        Embedding []float32 `json:"embedding"`
    }
    if err := json.Unmarshal(body, &ollamaResp); err != nil {
        log.Printf("Failed to parse Ollama response: %v", err)
        return fmt.Errorf("failed to parse embedding response: %v", err)
    }
    doc.Embedding = ollamaResp.Embedding

    // Insert into database (your provided function)
    embeddingStr := vectorToString(doc.Embedding)
    query := `INSERT INTO documents (content, media_type, file_name, embedding) VALUES ($1, $2, $3, $4) RETURNING id, created_at`
    log.Printf("Executing query: %s", query)
    log.Printf("Parameters: content=%s, media_type=%s, file_name=%s, embedding_length=%d",
        doc.Content, doc.MediaType, doc.FileName, len(doc.Embedding))
    err = db.QueryRow(query, doc.Content, doc.MediaType, doc.FileName, embeddingStr).Scan(&doc.ID, &doc.CreatedAt)
    if err != nil {
        log.Printf("Database error: %v", err)
        return fmt.Errorf("failed to save document: %v", err)
    }

    return nil
}
// func CreateDocument(db *sql.DB, doc *models.Document) error {
// 	embeddingStr := vectorToString(doc.Embedding)
//
// 	query := `INSERT INTO documents (content, media_type, file_name, embedding) VALUES ($1, $2, $3, $4) RETURNING id, created_at`
//
// 	log.Printf("Executing query: %s", query)
// 	log.Printf("Parameters: content=%s, media_type=%s, file_name=%s, embedding_length=%d", 
// 		doc.Content, doc.MediaType, doc.FileName, len(doc.Embedding))
//
// 	err := db.QueryRow(query, doc.Content, doc.MediaType, doc.FileName, embeddingStr).Scan(&doc.ID, &doc.CreatedAt)
// 	if err != nil {
// 		log.Printf("Database error: %v", err)
// 	}
// 	return err
// }

// Helper function to convert []float32 to pgvector string format
func vectorToString(embedding []float32) string {
	strValues := make([]string, len(embedding))
	for i, v := range embedding {
		strValues[i] = fmt.Sprintf("%f", v)
	}
	return "[" + strings.Join(strValues, ",") + "]"
}



// func CreateDocument(db *sql.DB, doc *models.Document) error {
// 	query := `INSERT INTO documents (content, media_type, file_name, embedding) VALUES ($1, $2, $3, $4) RETURNING id, created_at`
// 	err := db.QueryRow(query, doc.Content, doc.MediaType, doc.FileName, pq.Array(doc.Embedding)).Scan(&doc.ID, &doc.CreatedAt)
// 	return err
// }


// func CreateDocument(db *sql.DB, doc *models.Document) error {
//   query := `INSERT INTO documents (content, embedding) VALUES ($1, $2) RETURNING id, created_at`
// 	err := db.QueryRow(query, doc.Content, pq.Array(doc.Embedding)).Scan(&doc.ID, &doc.CreatedAt)
// 	return err
// }

func SearchSimilarDocuments(db *sql.DB, queryEmbedding []float32, limit int) ([]models.Document, error) {
	embeddingStr := vectorToString(queryEmbedding)

	query := `
		SELECT id, content, media_type, file_name, embedding, created_at,
		       1 - (embedding <=> $1) as similarity
		FROM documents 
		ORDER BY embedding <=> $1 
		LIMIT $2
	`

	log.Printf("Executing search query with embedding length: %d, limit: %d", len(queryEmbedding), limit)

	rows, err := db.Query(query, embeddingStr, limit)
	if err != nil {
		log.Printf("Search query error: %v", err)
		return nil, err
	}
	defer rows.Close()

	var documents []models.Document
	for rows.Next() {
		var doc models.Document
		var similarity float64
		var embeddingStr string
		var mediaType, fileName sql.NullString

		err := rows.Scan(&doc.ID, &doc.Content, &mediaType, &fileName, &embeddingStr, &doc.CreatedAt, &similarity)
		if err != nil {
			log.Printf("Row scan error: %v", err)
			continue
		}

		if mediaType.Valid {
			doc.MediaType = &mediaType.String
		}
		if fileName.Valid {
			doc.FileName = &fileName.String
		}
		
		doc.Embedding = parseVectorString(embeddingStr)

		log.Printf("Found document ID %d with similarity: %.4f", doc.ID, similarity)
		documents = append(documents, doc)
	}

	return documents, nil
}

func parseVectorString(vectorSTR string) []float32 {
	vectorSTR = strings.Trim(vectorSTR, "[]")
	if vectorSTR == "" {
		return []float32{}
	}

	parts := strings.Split(vectorSTR, ",")
	embedding := make([]float32, len(parts))
	
	for i, part := range  parts {
		if val, err := strconv.ParseFloat(strings.TrimSpace(part), 32); err == nil {
			embedding[i] = float32(val)
		}
	}

	return embedding
}
