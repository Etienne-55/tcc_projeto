package handlers

import (
	"bytes"
	"database/sql"
	"golang_crud/models"
	"golang_crud/services"
	"io"
	"log"
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
)


func CreateDocument(db *sql.DB) gin.HandlerFunc {
    return func(c *gin.Context) {
        var doc models.Document

        // Debug: Print raw body
        body, _ := c.GetRawData()
        log.Printf("Raw body: %s\n", string(body))

        // Reset body for binding
        c.Request.Body = io.NopCloser(bytes.NewBuffer(body))

        if err := c.ShouldBindJSON(&doc); err != nil {
            log.Printf("Binding error: %v\n", err)
            c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
            return
        }

        log.Printf("Parsed document: %+v\n", doc)

        if err := services.CreateDocument(db, &doc); err != nil {
            c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create document"})
            return
        }

        c.JSON(http.StatusCreated, gin.H{
            "message": "Document created successfully",
            "document": doc,
        })
    }
}

// func CreateDocument(db * sql.DB) gin.HandlerFunc {
// 	return func(c *gin.Context) {
// 		var doc models.Document
//
// 		if err := c.ShouldBind(&doc); err != nil {
// 			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
// 			return
//
// 			}
//
// 		if err := services.CreateDocument(db, &doc); err != nil {
// 			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create document"})
// 			return
// 		}
// 		c.JSON(http.StatusCreated, gin.H{
// 			"message": "Document created successfully",
// 			"document": doc,
//
// 		})
// 	}
// }

type DocumentHandlers struct {
	DB *sql.DB
}

func (h *DocumentHandlers) SearchDocuments(c *gin.Context) {
	query := c.Query("q")
	if query == "" {
		c.JSON(400, gin.H{"error": "Query parameter 'q' is required"})
		return
	}

	limitStr := c.DefaultQuery("limit", "5")
	limit, err := strconv.Atoi(limitStr)
	if err != nil {
		limit = 5
	}

	log.Printf("Searching for: %s (limit: %d)", query, limit)

	documents, err := services.SearchDocuments(h.DB, query, limit)
	if err != nil {
		log.Printf("Search error: %v", err)
		c.JSON(500, gin.H{"error": "Failed to search documents"})
		return
	}

	log.Printf("Found %d similar documents", len(documents))
	c.JSON(200, gin.H{
		"query": query,
		"results": documents,
		"count": len(documents),
	})
}



func (h *DocumentHandlers) GetAllDocuments(c *gin.Context) {
	query := `SELECT id, content, media_type, file_name, created_at FROM documents ORDER BY id DESC LIMIT 10`
	
	rows, err := h.DB.Query(query)
	if err != nil {
		c.JSON(500, gin.H{"error": "Database error"})
		return
	}
	defer rows.Close()
	
	var documents []models.Document
	for rows.Next() {
		var doc models.Document
		rows.Scan(&doc.ID, &doc.Content, &doc.MediaType, &doc.FileName, &doc.CreatedAt)
		documents = append(documents, doc)
	}
	
	c.JSON(200, gin.H{"documents": documents, "count": len(documents)})
}

