# Model Context Protocol (MCP) Servers - Complete Guide

Comprehensive coverage of Model Context Protocol servers for AI model integration and context management.

## 📚 Table of Contents

- [Overview](#overview)
- [What is Model Context Protocol?](#what-is-model-context-protocol)
- [MCP Architecture](#mcp-architecture)
- [Server Implementation](#server-implementation)
- [Client-Server Communication](#client-server-communication)
- [Use Cases & Applications](#use-cases--applications)
- [Best Practices](#best-practices)
- [Resources](#resources)

## 🎯 Overview

Model Context Protocol (MCP) is a standardized protocol for AI models to communicate with external data sources and tools. MCP servers act as intermediaries that provide structured access to various data sources, enabling AI models to retrieve relevant context and information dynamically.

## 🤖 What is Model Context Protocol?

Model Context Protocol is an open standard that defines how AI models can:
- **Query external data sources** in real-time
- **Access structured information** from databases, APIs, and tools
- **Maintain context** across multiple interactions
- **Integrate with various systems** seamlessly

### Key Benefits
- **Standardized Interface** - Consistent API across different data sources
- **Real-time Access** - Dynamic context retrieval during model inference
- **Security** - Controlled access to sensitive data
- **Scalability** - Support for multiple concurrent connections
- **Extensibility** - Easy addition of new data sources and tools

## 🏗️ MCP Architecture

### **Core Components**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Model     │    │   MCP Client    │    │   MCP Server    │
│                 │◄──►│                 │◄──►│                 │
│ - LLM          │    │ - Protocol      │    │ - Data Sources  │
│ - Agent        │    │ - Serialization │    │ - Tools         │
│ - Application  │    │ - Connection    │    │ - APIs          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Protocol Layers**

1. **Transport Layer** - HTTP/HTTPS, WebSocket, gRPC
2. **Message Layer** - JSON-RPC 2.0 compatible
3. **Schema Layer** - Structured data definitions
4. **Authentication Layer** - API keys, OAuth, certificates

## 🔧 Server Implementation

### **Basic MCP Server Structure**

```python
import asyncio
import json
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class MCPRequest:
    id: str
    method: str
    params: Dict[str, Any]

@dataclass
class MCPResponse:
    id: str
    result: Any = None
    error: Dict[str, Any] = None

class MCPServer:
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.resources = {}
        self.tools = {}
        self.initialized = False
    
    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle incoming MCP requests"""
        try:
            if request.method == "initialize":
                return await self.handle_initialize(request)
            elif request.method == "resources/list":
                return await self.handle_list_resources(request)
            elif request.method == "resources/read":
                return await self.handle_read_resource(request)
            elif request.method == "tools/list":
                return await self.handle_list_tools(request)
            elif request.method == "tools/call":
                return await self.handle_call_tool(request)
            else:
                return MCPResponse(
                    id=request.id,
                    error={"code": -32601, "message": f"Method {request.method} not found"}
                )
        except Exception as e:
            return MCPResponse(
                id=request.id,
                error={"code": -32603, "message": str(e)}
            )
    
    async def handle_initialize(self, request: MCPRequest) -> MCPResponse:
        """Handle initialization request"""
        self.initialized = True
        return MCPResponse(
            id=request.id,
            result={
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "resources": True,
                    "tools": True
                },
                "serverInfo": {
                    "name": self.name,
                    "version": self.version
                }
            }
        )
    
    async def handle_list_resources(self, request: MCPRequest) -> MCPResponse:
        """List available resources"""
        return MCPResponse(
            id=request.id,
            result={
                "resources": [
                    {
                        "uri": uri,
                        "name": info["name"],
                        "description": info["description"],
                        "mimeType": info["mime_type"]
                    }
                    for uri, info in self.resources.items()
                ]
            }
        )
    
    async def handle_read_resource(self, request: MCPRequest) -> MCPResponse:
        """Read a specific resource"""
        uri = request.params.get("uri")
        if uri not in self.resources:
            return MCPResponse(
                id=request.id,
                error={"code": -32602, "message": f"Resource {uri} not found"}
            )
        
        resource = self.resources[uri]
        return MCPResponse(
            id=request.id,
            result={
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": resource["mime_type"],
                        "text": resource["content"]
                    }
                ]
            }
        )
    
    async def handle_list_tools(self, request: MCPRequest) -> MCPResponse:
        """List available tools"""
        return MCPResponse(
            id=request.id,
            result={
                "tools": [
                    {
                        "name": name,
                        "description": info["description"],
                        "inputSchema": info["input_schema"]
                    }
                    for name, info in self.tools.items()
                ]
            }
        )
    
    async def handle_call_tool(self, request: MCPRequest) -> MCPResponse:
        """Call a specific tool"""
        name = request.params.get("name")
        arguments = request.params.get("arguments", {})
        
        if name not in self.tools:
            return MCPResponse(
                id=request.id,
                error={"code": -32602, "message": f"Tool {name} not found"}
            )
        
        try:
            tool = self.tools[name]
            result = await tool["function"](arguments)
            return MCPResponse(
                id=request.id,
                result={"content": result}
            )
        except Exception as e:
            return MCPResponse(
                id=request.id,
                error={"code": -32603, "message": f"Tool execution failed: {str(e)}"}
            )
    
    def add_resource(self, uri: str, name: str, description: str, content: str, mime_type: str = "text/plain"):
        """Add a resource to the server"""
        self.resources[uri] = {
            "name": name,
            "description": description,
            "content": content,
            "mime_type": mime_type
        }
    
    def add_tool(self, name: str, description: str, input_schema: Dict, function):
        """Add a tool to the server"""
        self.tools[name] = {
            "description": description,
            "input_schema": input_schema,
            "function": function
        }

# Example usage
async def example_tool(arguments: Dict) -> str:
    """Example tool that processes arguments"""
    name = arguments.get("name", "World")
    return f"Hello, {name}!"

# Create and configure server
server = MCPServer("Example MCPServer", "1.0.0")

# Add resources
server.add_resource(
    uri="file:///example.txt",
    name="Example File",
    description="A sample text file",
    content="This is an example file content.",
    mime_type="text/plain"
)

# Add tools
server.add_tool(
    name="greet",
    description="A simple greeting tool",
    input_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name to greet"}
        }
    },
    function=example_tool
)
```

### **HTTP Server Implementation**

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="MCP Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global server instance
mcp_server = MCPServer("HTTP MCPServer", "1.0.0")

@app.post("/mcp")
async def handle_mcp_request(request: Dict[str, Any]):
    """Handle MCP requests via HTTP"""
    try:
        mcp_request = MCPRequest(
            id=request.get("id"),
            method=request.get("method"),
            params=request.get("params", {})
        )
        
        response = await mcp_server.handle_request(mcp_request)
        return {
            "jsonrpc": "2.0",
            "id": response.id,
            "result": response.result,
            "error": response.error
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "server": mcp_server.name}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### **WebSocket Server Implementation**

```python
import asyncio
import websockets
import json
from typing import Set

class WebSocketMCPServer:
    def __init__(self, mcp_server: MCPServer):
        self.mcp_server = mcp_server
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
    
    async def register(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new client"""
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
    
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle individual client connections"""
        await self.register(websocket)
        
        try:
            async for message in websocket:
                try:
                    # Parse JSON message
                    data = json.loads(message)
                    
                    # Create MCP request
                    mcp_request = MCPRequest(
                        id=data.get("id"),
                        method=data.get("method"),
                        params=data.get("params", {})
                    )
                    
                    # Handle request
                    response = await self.mcp_server.handle_request(mcp_request)
                    
                    # Send response
                    response_data = {
                        "jsonrpc": "2.0",
                        "id": response.id
                    }
                    
                    if response.result is not None:
                        response_data["result"] = response.result
                    if response.error is not None:
                        response_data["error"] = response.error
                    
                    await websocket.send(json.dumps(response_data))
                    
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": "Parse error"}
                    }))
                except Exception as e:
                    await websocket.send(json.dumps({
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32603, "message": str(e)}
                    }))
        except websockets.exceptions.ConnectionClosed:
            pass

async def start_websocket_server(host: str = "localhost", port: int = 8765):
    """Start WebSocket MCP server"""
    mcp_server = MCPServer("WebSocket MCPServer", "1.0.0")
    ws_server = WebSocketMCPServer(mcp_server)
    
    # Add some example resources and tools
    mcp_server.add_resource(
        uri="file:///example.txt",
        name="Example File",
        description="A sample text file",
        content="This is an example file content."
    )
    
    async def example_tool(arguments: Dict) -> str:
        name = arguments.get("name", "World")
        return f"Hello, {name}!"
    
    mcp_server.add_tool(
        name="greet",
        description="A simple greeting tool",
        input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        function=example_tool
    )
    
    async with websockets.serve(ws_server.handle_client, host, port):
        print(f"WebSocket MCP server running on ws://{host}:{port}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(start_websocket_server())
```

## 🔄 Client-Server Communication

### **MCP Client Implementation**

```python
import asyncio
import json
import aiohttp
from typing import Dict, Any, List

class MCPClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session = None
        self.request_id = 0
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _get_next_id(self) -> str:
        """Get next request ID"""
        self.request_id += 1
        return str(self.request_id)
    
    async def _send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send request to MCP server"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        request_data = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": method,
            "params": params or {}
        }
        
        async with self.session.post(
            f"{self.server_url}/mcp",
            json=request_data,
            headers={"Content-Type": "application/json"}
        ) as response:
            result = await response.json()
            
            if "error" in result:
                raise Exception(f"MCP Error: {result['error']['message']}")
            
            return result.get("result", {})
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize connection with MCP server"""
        return await self._send_request("initialize")
    
    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources"""
        result = await self._send_request("resources/list")
        return result.get("resources", [])
    
    async def read_resource(self, uri: str) -> str:
        """Read a specific resource"""
        result = await self._send_request("resources/read", {"uri": uri})
        contents = result.get("contents", [])
        if contents:
            return contents[0].get("text", "")
        return ""
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        result = await self._send_request("tools/list")
        return result.get("tools", [])
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a specific tool"""
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })
        return result.get("content")

# Example client usage
async def example_client_usage():
    async with MCPClient("http://localhost:8000") as client:
        # Initialize connection
        server_info = await client.initialize()
        print(f"Connected to: {server_info['serverInfo']['name']} v{server_info['serverInfo']['version']}")
        
        # List resources
        resources = await client.list_resources()
        print(f"Available resources: {len(resources)}")
        for resource in resources:
            print(f"  - {resource['name']}: {resource['description']}")
        
        # Read a resource
        if resources:
            content = await client.read_resource(resources[0]['uri'])
            print(f"Resource content: {content}")
        
        # List tools
        tools = await client.list_tools()
        print(f"Available tools: {len(tools)}")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")
        
        # Call a tool
        if tools:
            result = await client.call_tool(tools[0]['name'], {"name": "Alice"})
            print(f"Tool result: {result}")

# Run example
if __name__ == "__main__":
    asyncio.run(example_client_usage())
```

## 🚀 Use Cases & Applications

### **1. AI Model Integration**
- **Context Retrieval** - Get relevant information during inference
- **Dynamic Knowledge** - Access up-to-date data sources
- **Multi-modal Support** - Handle text, images, and structured data

### **2. Enterprise Applications**
- **Document Management** - Access internal knowledge bases
- **Database Integration** - Query business data securely
- **API Aggregation** - Combine multiple external services

### **3. Development Tools**
- **Code Context** - Access repository information
- **Documentation** - Retrieve relevant docs and examples
- **Testing** - Integrate with testing frameworks

### **4. Research & Education**
- **Academic Databases** - Access research papers and datasets
- **Interactive Learning** - Dynamic content delivery
- **Collaboration** - Share resources across institutions

## 💡 Best Practices

### **1. Security Considerations**
```python
class SecureMCPServer(MCPServer):
    def __init__(self, name: str, version: str, api_key: str = None):
        super().__init__(name, version)
        self.api_key = api_key
        self.rate_limits = {}
    
    async def authenticate_request(self, headers: Dict[str, str]) -> bool:
        """Authenticate incoming requests"""
        if not self.api_key:
            return True  # No authentication required
        
        auth_header = headers.get("Authorization")
        if not auth_header:
            return False
        
        # Check API key
        if auth_header.startswith("Bearer "):
            key = auth_header[7:]
            return key == self.api_key
        
        return False
    
    async def check_rate_limit(self, client_id: str) -> bool:
        """Check rate limiting for client"""
        import time
        current_time = time.time()
        
        if client_id not in self.rate_limits:
            self.rate_limits[client_id] = []
        
        # Remove old requests (older than 1 minute)
        self.rate_limits[client_id] = [
            req_time for req_time in self.rate_limits[client_id]
            if current_time - req_time < 60
        ]
        
        # Check if limit exceeded (100 requests per minute)
        if len(self.rate_limits[client_id]) >= 100:
            return False
        
        # Add current request
        self.rate_limits[client_id].append(current_time)
        return True
```

### **2. Error Handling**
```python
class RobustMCPServer(MCPServer):
    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle requests with robust error handling"""
        try:
            # Validate request
            if not request.id or not request.method:
                return MCPResponse(
                    id=request.id or "unknown",
                    error={"code": -32600, "message": "Invalid Request"}
                )
            
            # Check if server is initialized
            if request.method != "initialize" and not self.initialized:
                return MCPResponse(
                    id=request.id,
                    error={"code": -32002, "message": "Server not initialized"}
                )
            
            # Handle request
            return await super().handle_request(request)
            
        except Exception as e:
            # Log error for debugging
            print(f"Error handling request {request.id}: {str(e)}")
            
            return MCPResponse(
                id=request.id,
                error={"code": -32603, "message": "Internal error"}
            )
```

### **3. Performance Optimization**
```python
class CachedMCPServer(MCPServer):
    def __init__(self, name: str, version: str, cache_ttl: int = 300):
        super().__init__(name, version)
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.cache_timestamps = {}
    
    def _get_cache_key(self, method: str, params: Dict) -> str:
        """Generate cache key for request"""
        return f"{method}:{hash(str(sorted(params.items())))}"
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached item is still valid"""
        import time
        if key not in self.cache_timestamps:
            return False
        
        return time.time() - self.cache_timestamps[key] < self.cache_ttl
    
    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle requests with caching"""
        # Check cache for read operations
        if request.method in ["resources/read", "tools/list", "resources/list"]:
            cache_key = self._get_cache_key(request.method, request.params)
            
            if cache_key in self.cache and self._is_cache_valid(cache_key):
                return MCPResponse(
                    id=request.id,
                    result=self.cache[cache_key]
                )
        
        # Handle request normally
        response = await super().handle_request(request)
        
        # Cache successful responses
        if response.result and request.method in ["resources/read", "tools/list", "resources/list"]:
            cache_key = self._get_cache_key(request.method, request.params)
            self.cache[cache_key] = response.result
            self.cache_timestamps[cache_key] = time.time()
        
        return response
```

## 📚 Resources

### **Official Documentation**
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [MCP GitHub Repository](https://github.com/modelcontextprotocol)
- [MCP Examples](https://github.com/modelcontextprotocol/python)

### **Blogs & Articles**
- [Introducing the Model Context Protocol](https://blog.anthropic.com/introducing-the-model-context-protocol)
- [Building MCP Servers](https://modelcontextprotocol.io/docs/build)

### **Python Libraries**
- [mcp](https://pypi.org/project/mcp/) - Official MCP Python library
- [fastapi](https://fastapi.tiangolo.com/) - Modern web framework for APIs
- [websockets](https://websockets.readthedocs.io/) - WebSocket library
- [aiohttp](https://docs.aiohttp.org/) - Async HTTP client/server

### **Development Tools**
- [MCP Server](https://github.com/modelcontextprotocol/servers)
- [MCP Client Examples](https://github.com/modelcontextprotocol/python/tree/main/examples)
- [Testing Framework](https://github.com/modelcontextprotocol/python/tree/main/tests)

---

**Happy MCP Server Development! 🚀✨**

*Model Context Protocol servers enable AI models to access dynamic, real-time information from various sources, making them more powerful and contextually aware.*

