#!/bin/bash

# Test script for the HackRx API
# Make sure to replace YOUR_BEARER_TOKEN with a valid token
# and set a valid OpenAI API key in the .env file

echo "Testing HackRx API..."

# Test health endpoint
echo "1. Testing health endpoint:"
curl -X GET http://localhost:8000/health
echo -e "\n"

# Test root endpoint  
echo "2. Testing root endpoint:"
curl -X GET http://localhost:8000/
echo -e "\n"

# Test the main endpoint (uncomment and modify when you have a valid API key)
echo "3. Testing main endpoint (requires valid OpenAI API key):"
echo "curl -X POST http://localhost:8000/hackrx/run \\"
echo "  -H \"Authorization: Bearer YOUR_BEARER_TOKEN\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"documents\":\"https://example.com/sample.pdf\",\"questions\":[\"What is this document about?\"]}'"
echo ""
echo "Note: Replace YOUR_BEARER_TOKEN with a valid bearer token and ensure OpenAI API key is set in .env file"
