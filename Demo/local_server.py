import http.server
import socketserver
import urllib.request
import urllib.parse
import json
import os

PORT = 8000

def fetch_reddit_pullpush(subreddit, limit):
    # Cap size at 100 to avoid API limits and ensure stability
    size = min(limit, 100)
    url = f"https://api.pullpush.io/reddit/search/submission/?subreddit={subreddit}&size={size}"
    print(f"[Proxy] Fetching Reddit data via PullPush API: {url}")
    
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode())
            items = res_data.get("data", [])
            children = []
            
            for item in items:
                mapped = {
                    "id": item.get("id", ""),
                    "title": item.get("title", ""),
                    "selftext": item.get("selftext", ""),
                    "author": item.get("author", "anonymous"),
                    "score": item.get("score", 0),
                    "permalink": item.get("permalink", "")
                }
                children.append({"data": mapped})
                
            print(f"[Proxy] Successfully fetched and parsed {len(children)} Reddit posts.")
            return {"data": {"children": children}}
            
    except Exception as e:
        print(f"[Proxy] Error fetching from PullPush API: {e}")
        return {"data": {"children": []}, "error": str(e)}

class LocalDemoHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        
        # Intercept Reddit API request
        if parsed_url.path == '/api/reddit':
            query_params = urllib.parse.parse_qs(parsed_url.query)
            subreddit = query_params.get('subreddit', ['TrueOffMyChest'])[0]
            limit = int(query_params.get('limit', ['40'])[0])
            
            # Fetch from PullPush
            reddit_data = fetch_reddit_pullpush(subreddit, limit)
            
            # Send HTTP headers
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Write response
            self.wfile.write(json.dumps(reddit_data).encode())
        else:
            # Serve local static files (index.html, script.js, style.css)
            super().do_GET()

# Change working directory to where this file is located to serve files correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Allow socket address reuse to prevent "Address already in use" errors on restart
socketserver.TCPServer.allow_reuse_address = True

with socketserver.TCPServer(("", PORT), LocalDemoHandler) as httpd:
    print("=" * 60)
    print(f"ENELPI LOCAL DEMO SERVER RUNNING AT: http://localhost:{PORT}")
    print("Serving static files and local Reddit proxy via PullPush API (real Reddit data & links)")
    print("Press CTRL+C to stop the server")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local server...")
