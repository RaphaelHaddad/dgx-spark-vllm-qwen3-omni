#!/usr/bin/env python3
"""
Proxy de debug pour voir exactement ce que Chatbox envoie
Lance sur le port 8001 et forward vers 8000
"""
from flask import Flask, request, Response
import requests
import json

app = Flask(__name__)
TARGET_URL = "http://localhost:8000"

@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
def proxy(path):
    url = f"{TARGET_URL}/{path}"
    
    print(f"\n{'='*60}")
    print(f"📨 {request.method} /{path}")
    print(f"{'='*60}")
    
    # Headers
    print("\n📋 Headers:")
    for key, value in request.headers:
        if key.lower() not in ['host', 'connection']:
            print(f"  {key}: {value}")
    
    # Body
    if request.method in ['POST', 'PUT', 'PATCH']:
        try:
            body = request.get_json()
            print("\n📝 Body (Request):")
            print(json.dumps(body, indent=2, ensure_ascii=False))
        except:
            print("\n📝 Body (Raw):")
            print(request.get_data(as_text=True))
    
    # Forward request
    try:
        resp = requests.request(
            method=request.method,
            url=url,
            headers={key: value for (key, value) in request.headers if key.lower() not in ['host', 'connection']},
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False
        )
        
        print("\n📬 Response:")
        print(f"  Status: {resp.status_code}")
        
        try:
            response_json = resp.json()
            print("  Body:")
            print(json.dumps(response_json, indent=2, ensure_ascii=False)[:500])
            
            # Vérifier le content
            if 'choices' in response_json:
                content = response_json['choices'][0]['message']['content']
                print(f"\n✅ Content extrait: '{content[:100]}...'")
                print(f"   Longueur: {len(content)} caractères")
        except:
            print(f"  Body (Raw): {resp.text[:200]}...")
        
        print(f"{'='*60}\n")
        
        # Return response
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        headers = [(name, value) for (name, value) in resp.raw.headers.items()
                   if name.lower() not in excluded_headers]
        
        return Response(resp.content, resp.status_code, headers)
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        print(f"{'='*60}\n")
        return {"error": str(e)}, 500

if __name__ == '__main__':
    print("🔍 Proxy de debug démarré")
    print("📍 Écoute sur: http://localhost:8001")
    print("🎯 Forward vers: http://localhost:8000")
    print("")
    print("👉 Configurez Chatbox pour utiliser:")
    print("   http://spark-c63e.local:8001/v1")
    print("")
    app.run(host='0.0.0.0', port=8001, debug=False)
