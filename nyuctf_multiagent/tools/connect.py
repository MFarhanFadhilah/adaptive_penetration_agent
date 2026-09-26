"""
Network Connect Tools for CTF Agent
Assignment 4 - Interactive network connection tools
Inspired by EnIGMA agent implementation
"""
import socket
import select
import threading
from queue import Queue, Empty

from ..logging import logger
from .tool import Tool


class NetworkConnection:
    """
    Manages a persistent network connection with buffered I/O.
    This is a shared singleton that tools interact with.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.socket = None
        self.connected = False
        self.receive_buffer = ""
        self.receive_queue = Queue()
        self.receive_thread = None
        self.host = None
        self.port = None
        self._initialized = True
    
    def connect(self, host, port, timeout=10):
        """Establish connection to server"""
        if self.connected:
            return {"error": f"Already connected to {self.host}:{self.port}. Disconnect first."}
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(timeout)
            self.socket.connect((host, int(port)))
            self.socket.setblocking(False)
            self.connected = True
            self.host = host
            self.port = port
            self.receive_buffer = ""
            
            # Start background receive thread
            self._start_receive_thread()
            
            return {"success": True, "message": f"Connected to {host}:{port}"}
        except socket.timeout:
            return {"error": f"Connection to {host}:{port} timed out"}
        except ConnectionRefusedError:
            return {"error": f"Connection to {host}:{port} refused"}
        except socket.gaierror as e:
            return {"error": f"Failed to resolve host {host}: {str(e)}"}
        except Exception as e:
            return {"error": f"Connection failed: {str(e)}"}
    
    def _start_receive_thread(self):
        """Start background thread to receive data"""
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()
    
    def _receive_loop(self):
        """Background thread to continuously receive data"""
        while self.connected and self.socket:
            try:
                ready, _, _ = select.select([self.socket], [], [], 0.1)
                if ready:
                    data = self.socket.recv(4096)
                    if data:
                        decoded = data.decode('utf-8', errors='replace')
                        self.receive_queue.put(decoded)
                    else:
                        # Connection closed by server
                        self.connected = False
                        break
            except (socket.error, OSError):
                break
            except Exception:
                break
    
    def disconnect(self):
        """Close the connection"""
        if not self.connected:
            return {"error": "No active connection to disconnect"}
        
        try:
            old_host = self.host
            old_port = self.port
            self.connected = False
            if self.socket:
                self.socket.close()
            self.socket = None
            self.host = None
            self.port = None
            self.receive_buffer = ""
            # Clear the queue
            while not self.receive_queue.empty():
                try:
                    self.receive_queue.get_nowait()
                except Empty:
                    break
            return {"success": True, "message": f"Disconnected from {old_host}:{old_port}"}
        except Exception as e:
            return {"error": f"Error disconnecting: {str(e)}"}
    
    def readline(self, timeout=5):
        """Read a single line from the connection"""
        if not self.connected:
            return {"error": "No active connection. Use connect_to_server first."}
        
        import time
        start_time = time.time()
        
        # First check buffer for complete line
        while True:
            if '\n' in self.receive_buffer:
                line, self.receive_buffer = self.receive_buffer.split('\n', 1)
                return {"line": line.rstrip('\r'), "success": True}
            
            # Check timeout
            if time.time() - start_time > timeout:
                if self.receive_buffer:
                    # Return partial data if timeout
                    partial = self.receive_buffer
                    self.receive_buffer = ""
                    return {"line": partial, "partial": True, "success": True}
                return {"error": "Timeout waiting for data", "timeout": True}
            
            # Try to get more data from queue
            try:
                data = self.receive_queue.get(timeout=0.1)
                self.receive_buffer += data
            except Empty:
                continue
    
    def sendline(self, line):
        """Send a line to the connection"""
        if not self.connected:
            return {"error": "No active connection. Use connect_to_server first."}
        
        try:
            # Add newline if not present
            if not line.endswith('\n'):
                line = line + '\n'
            self.socket.sendall(line.encode('utf-8'))
            return {"success": True, "message": f"Sent: {line.strip()}"}
        except socket.error as e:
            self.connected = False
            return {"error": f"Send failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Send error: {str(e)}"}
    
    def read_all_available(self, timeout=1):
        """Read all available data without blocking"""
        if not self.connected:
            return {"error": "No active connection"}
        
        import time
        start_time = time.time()
        all_data = self.receive_buffer
        self.receive_buffer = ""
        
        while time.time() - start_time < timeout:
            try:
                data = self.receive_queue.get(timeout=0.1)
                all_data += data
            except Empty:
                if all_data:
                    break
                continue
        
        if all_data:
            return {"data": all_data, "success": True}
        return {"data": "", "success": True, "message": "No data available"}


# Global connection instance
_connection = NetworkConnection()


class ConnectToServerTool(Tool):
    """Tool to establish a network connection to a server"""
    NAME = "connect_to_server"
    DESCRIPTION = "Establish a TCP network connection to a CTF challenge server. Use this for interactive challenges like pwn/nc connections."
    PARAMETERS = {
        "host": ("string", "The server hostname or IP address to connect to"),
        "port": ("number", "The port number to connect to"),
        "timeout": ("number", "Connection timeout in seconds (default: 10)")
    }
    REQUIRED_PARAMETERS = {"host", "port"}

    def __init__(self, environment):
        super().__init__()
        self.environment = environment

    def call(self, host=None, port=None, timeout=10):
        if host is None:
            return {"error": "Host not provided"}
        if port is None:
            return {"error": "Port not provided"}
        
        return _connection.connect(host, int(port), timeout)

    def print_tool_call(self, tool_call):
        host = tool_call.parsed_arguments.get('host', 'unknown')
        port = tool_call.parsed_arguments.get('port', 'unknown')
        logger.assistant_action(f"**{self.NAME}**: Connecting to {host}:{port}")
    
    def print_result(self, tool_result):
        if "error" in tool_result.result:
            logger.print(f"[bold]{self.NAME}[/bold]: [red]{tool_result.result['error']}[/red]", markup=True)
        else:
            logger.observation_message(f"**{self.NAME}**: {tool_result.result.get('message', 'Connected')}")


class ConnectDisconnectTool(Tool):
    """Tool to disconnect from the current network connection"""
    NAME = "connect_disconnect"
    DESCRIPTION = "Close the current network connection. Use this when done interacting with the server."
    PARAMETERS = {}
    REQUIRED_PARAMETERS = set()

    def __init__(self, environment):
        super().__init__()
        self.environment = environment

    def call(self):
        return _connection.disconnect()

    def print_tool_call(self, tool_call):
        logger.assistant_action(f"**{self.NAME}**: Disconnecting...")
    
    def print_result(self, tool_result):
        if "error" in tool_result.result:
            logger.print(f"[bold]{self.NAME}[/bold]: [red]{tool_result.result['error']}[/red]", markup=True)
        else:
            logger.observation_message(f"**{self.NAME}**: {tool_result.result.get('message', 'Disconnected')}")


class ConnectReadlineTool(Tool):
    """Tool to read a line from the network connection"""
    NAME = "connect_readline"
    DESCRIPTION = "Read a single line from the network connection. Returns the received line. Useful for reading prompts, responses, or flags from the server."
    PARAMETERS = {
        "timeout": ("number", "Timeout in seconds to wait for data (default: 5)")
    }
    REQUIRED_PARAMETERS = set()

    def __init__(self, environment):
        super().__init__()
        self.environment = environment

    def call(self, timeout=5):
        return _connection.readline(timeout)

    def print_tool_call(self, tool_call):
        timeout = tool_call.parsed_arguments.get('timeout', 5)
        logger.assistant_action(f"**{self.NAME}**: Reading line (timeout: {timeout}s)")
    
    def print_result(self, tool_result):
        if "error" in tool_result.result:
            logger.print(f"[bold]{self.NAME}[/bold]: [red]{tool_result.result['error']}[/red]", markup=True)
        else:
            line = tool_result.result.get('line', '')
            partial = tool_result.result.get('partial', False)
            status = "(partial)" if partial else ""
            logger.observation_message(f"**{self.NAME}** {status}:\n```\n{line}\n```")


class ConnectSendlineTool(Tool):
    """Tool to send a line to the network connection"""
    NAME = "connect_sendline"
    DESCRIPTION = "Send a line of text to the network connection. Automatically appends a newline. Use this to send commands, payloads, or responses to the server."
    PARAMETERS = {
        "line": ("string", "The line of text to send to the server")
    }
    REQUIRED_PARAMETERS = {"line"}

    def __init__(self, environment):
        super().__init__()
        self.environment = environment

    def call(self, line=None):
        if line is None:
            return {"error": "Line not provided"}
        return _connection.sendline(line)

    def print_tool_call(self, tool_call):
        line = tool_call.parsed_arguments.get('line', '')
        # Truncate for display if too long
        display_line = line if len(line) <= 100 else line[:100] + "..."
        logger.assistant_action(f"**{self.NAME}**: Sending `{display_line}`")
    
    def print_result(self, tool_result):
        if "error" in tool_result.result:
            logger.print(f"[bold]{self.NAME}[/bold]: [red]{tool_result.result['error']}[/red]", markup=True)
        else:
            logger.observation_message(f"**{self.NAME}**: {tool_result.result.get('message', 'Sent')}")


class ConnectReadAllTool(Tool):
    """Tool to read all available data from the connection"""
    NAME = "connect_read_all"
    DESCRIPTION = "Read all available data from the network connection without blocking. Useful for reading multi-line responses or banners."
    PARAMETERS = {
        "timeout": ("number", "Maximum time to wait for data (default: 1)")
    }
    REQUIRED_PARAMETERS = set()

    def __init__(self, environment):
        super().__init__()
        self.environment = environment

    def call(self, timeout=1):
        return _connection.read_all_available(timeout)

    def print_tool_call(self, tool_call):
        timeout = tool_call.parsed_arguments.get('timeout', 1)
        logger.assistant_action(f"**{self.NAME}**: Reading all available data (timeout: {timeout}s)")
    
    def print_result(self, tool_result):
        if "error" in tool_result.result:
            logger.print(f"[bold]{self.NAME}[/bold]: [red]{tool_result.result['error']}[/red]", markup=True)
        else:
            data = tool_result.result.get('data', '')
            if data:
                logger.observation_message(f"**{self.NAME}**:\n```\n{data}\n```")
            else:
                logger.observation_message(f"**{self.NAME}**: No data available")

    def teardown(self, ex_type, ex_val, tb):
        """Cleanup: disconnect when environment tears down"""
        if _connection.connected:
            _connection.disconnect()
