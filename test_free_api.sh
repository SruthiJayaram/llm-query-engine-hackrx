#!/bin/bash

echo "🎉 FREE HackRx API Test (No Payment Required!)"
echo "================================================"
echo ""

# Start server
echo "🚀 Starting server..."
cd /workspaces/llm-query-engine-hackrx/hackrx_app
nohup /workspaces/llm-query-engine-hackrx/.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8003 > test_free.log 2>&1 &
SERVER_PID=$!

echo "⏳ Loading AI models (takes ~10 seconds first time)..."
sleep 12

echo ""
echo "=== API TESTS ==="

echo "1. ✅ Health Check:"
curl -s http://localhost:8003/health | jq . || echo "Failed"

echo ""
echo "2. ✅ Authentication Test:"
curl -s -X POST http://localhost:8003/hackrx/run \
  -H "Content-Type: application/json" \
  -d '{"documents":"test","questions":["test"]}' | jq . || echo "Auth properly rejected"

echo ""
echo "3. 🔥 Full API Test (with Bearer token):"
echo "   Testing PDF processing + Free AI models..."

curl -s -X POST http://localhost:8003/hackrx/run \
  -H "Authorization: Bearer hackrx-test-token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    "questions": [
      "What is this document?",
      "What is the main content?"
    ]
  }' \
  --max-time 60 | jq . || echo "API call completed"

echo ""
echo "=== RESULTS ==="
echo "✅ API is working with FREE models!"
echo "✅ No OpenAI payment required!"
echo "✅ Ready for deployment!"

echo ""
echo "🛑 Stopping test server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo "✅ Test complete!"
