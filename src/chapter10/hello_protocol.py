from hello_agents.tools import MCPTool, A2ATool, ANPTool

# 1. MCP
mcp_tool = MCPTool()
result = mcp_tool.run({"action": "call_tool", "tool_name": "add", "arguments": {"a": 10, "b": 20}, })
print(f"MCP打印结果{result}")

# 2. A2A
a2a_tool = A2ATool(agent_url="http://localhost:5000")

# 3. ANP
anp_tool = ANPTool()
anp_tool.run({
    "action": "register_service",
    "service_id": "calculator",
    "service_type": "math",
    "endpoint": "http://localhost:8080"
})
services = anp_tool.run({"action": "discover_services", })
print(f"发现的服务: {services}")