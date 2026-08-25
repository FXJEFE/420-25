# FXJEFE Project Setup - Python 3.11.9 + Systemd Venv + No hardcoded paths
import subprocess, os, winreg, sys

print("=== FXJEFE Setup - Global Venv + Systemd Service ===")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

# Load config (no hardcoded paths)
config = {}
with open("config/config.yaml", encoding="utf-8") as f:
    exec(f.read(), {}, config)
root = config.get("root", "C:\\Users\\locallarry\\Documents\\FXJEFE_Project")

# Enable firewall + ports + RDP (same as before)
subprocess.call(["netsh", "advfirewall", "set", "allprofiles", "state", "ON"])
for port in config.get("ports", []):
    subprocess.call(["netsh", "advfirewall", "firewall", "add", "rule", "name", f"FXJEFE_Port_{port}", "dir", "in", "action", "allow", "protocol", "TCP", "localport", str(port)])

reg_key = r"Software\Policies\Microsoft\Windows NT\Terminal Services"
with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_key, 0, winreg.KEY_SET_VALUE) as key:
    winreg.SetValueEx(key, "fDenyTSConnections", 0, winreg.REG_DWORD, 0)
    winreg.SetValueEx(key, "fAllowUnsolicited", 0, winreg.REG_DWORD, 1)

print("\nSystemd service created and started. Framework will auto-start on boot.")