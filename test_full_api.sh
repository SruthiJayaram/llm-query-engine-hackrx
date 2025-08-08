#!/bin/bash

# Simple API test script
echo "🚀 Starting HackRx API Test..."

# Start server in background
cd /workspaces/llm-query-engine-hackrx/hackrx_app
/workspaces/llm-query-engine-hackrx/.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8001 > test_server.log 2>&1 &
SERVER_PID=$!

echo "⏳ Waiting for server to start..."
sleep 5

echo ""
echo "=== API STATUS CHECKS ==="
echo ""

# Test 1: Health
echo "1. Health Check:"
curl -s http://localhost:8001/health && echo ""

# Test 2: Root
echo "2. Root Endpoint:"
curl -s http://localhost:8001/ && echo ""

# Test 3: Auth failure (expected)
echo "3. Auth Test (no token - should fail):"
curl -s -X POST http://localhost:8001/hackrx/run -H "Content-Type: application/json" -d '{"documents":"test","questions":["test"]}' && echo ""

# Test 4: With token
echo "4. Main Endpoint (with token):"
echo "   This will test PDF processing + OpenAI integration..."
curl -s -X POST http://localhost:8001/hackrx/run \
  -H "Authorization: Bearer test-token-123" \
  -H "Content-Type: application/json" \
  -d '{"documents":"https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf","questions":["What is this document about?"]}' \
  --max-time 30 && echo ""

echo ""
echo "=== TEST COMPLETE ==="
echo "Stopping server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
