import os
import random

import requests
import yaml

LEVEL_ID = 0
TASK_ID = 1

TASK = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""

    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def read_server_endpoints():
    """Read server endpoints from server_info.yaml.

    Supports both formats:
    - New format: servers: [{ip, port}, ...]
    - Legacy format: server_ip/server_port
    """
    config_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(config_dir, "server_info.yaml")

    default_servers = [{"ip": "localhost", "port": 8000}]
    default_lb = "random"
    default_timeout = 3000

    if not os.path.exists(config_path):
        print(f"[Client] server_info.yaml not found at {config_path}")
        print(f"[Client] Using default server: {default_servers[0]['ip']}:{default_servers[0]['port']}")
        return default_servers, default_lb, default_timeout

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}

        raw_servers = config.get("servers")
        servers = []
        if isinstance(raw_servers, list):
            for item in raw_servers:
                if not isinstance(item, dict):
                    continue
                ip = item.get("ip") or item.get("host")
                port = item.get("port", 8000)
                if ip is None:
                    continue
                try:
                    servers.append({"ip": str(ip), "port": int(port)})
                except Exception:
                    continue

        if not servers:
            ip = config.get("server_ip", default_servers[0]["ip"])
            port = config.get("server_port", default_servers[0]["port"])
            try:
                servers = [{"ip": str(ip), "port": int(port)}]
            except Exception:
                servers = default_servers

        load_balance = str(config.get("load_balance", default_lb))
        if load_balance != "random":
            load_balance = default_lb

        request_timeout = config.get("request_timeout", default_timeout)
        try:
            request_timeout = int(request_timeout)
        except Exception:
            request_timeout = default_timeout

        pretty = ", ".join(f"{s['ip']}:{s['port']}" for s in servers)
        print(
            f"[Client] Loaded server config: [{pretty}] "
            f"(load_balance={load_balance}, timeout={request_timeout})"
        )
        return servers, load_balance, request_timeout

    except Exception as e:
        print(f"[Client] Error reading server_info.yaml: {e}")
        print(f"[Client] Using default server: {default_servers[0]['ip']}:{default_servers[0]['port']}")
        return default_servers, default_lb, default_timeout


def pick_server(servers, load_balance):
    if not servers:
        return {"ip": "localhost", "port": 8000}

    if load_balance == "random":
        return random.choice(servers)
    else:
        raise ValueError(f"Invalid load balance mode: {load_balance}")


def evaluate(program_path):
    """Evaluate a program by sending it to evaluation server(s)."""
    servers, load_balance, request_timeout = read_server_endpoints()

    program_src = read_file(program_path)
    if not program_src:
        return {
            "compiled": False,
            "correctness": False,
            "error": f"Failed to read program from {program_path}",
            "error_name": "File Read Error",
            "combined_score": 0.0,
        }

    request_data = {
        "program_src": program_src,
        "level_id": LEVEL_ID,
        "task_id": TASK_ID,
        "task": TASK,
    }

    if not servers:
        return {
            "compiled": False,
            "correctness": False,
            "error": "No valid servers configured",
            "error_name": "No Server Config",
            "combined_score": 0.0,
        }

    attempt_count = len(servers)
    tried = []
    failed = set()

    for _ in range(attempt_count):
        available = [s for s in servers if (s["ip"], s["port"]) not in failed]
        if not available:
            break
        server = pick_server(available, load_balance)

        server_url = f"http://{server['ip']}:{server['port']}/evaluate"
        tried.append(f"{server['ip']}:{server['port']}")

        try:
            print(f"[Client] Connecting to evaluation server at {server_url}")
            print("[Client] Sending evaluation request...")

            response = requests.post(server_url, json=request_data, timeout=request_timeout)
            print(f"[Client] Received response (status code: {response.status_code})")

            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("success") and response_data.get("result"):
                    print("  - Success: True")
                    print(f"  - Device ID: {response_data.get('device_id')}")
                    print(f"  - Compiled: {response_data.get('result', {}).get('compiled')}")
                    print(f"  - Correctness: {response_data.get('result', {}).get('correctness')}")
                    print(f"  - Combined Score: {response_data.get('result', {}).get('combined_score', 0):.6f}")
                    return response_data["result"]

                print("  - Success: False")
                print(f"  - Response data: {response_data.get('result', {}).get('metadata', {})}")
                return response_data["result"]

            print(f"  - HTTP Error from {server_url}: {response.status_code}")
            failed.add((server["ip"], server["port"]))

        except requests.exceptions.ConnectionError:
            print(f"[Client] Connection refused: {server_url}")
            failed.add((server["ip"], server["port"]))
        except requests.exceptions.Timeout:
            print(f"[Client] Request timeout: {server_url}")
            failed.add((server["ip"], server["port"]))
        except Exception as e:
            print(f"[Client] Unexpected error on {server_url}: {e}")
            failed.add((server["ip"], server["port"]))
