import os
import sys
import importlib.util

base_dir = os.path.dirname(os.path.abspath(__file__))
kewords_dir = os.path.join(base_dir, "Kewords")
if kewords_dir not in sys.path:
    sys.path.insert(0, kewords_dir)

spec = importlib.util.spec_from_file_location("kewords_app", os.path.join(kewords_dir, "app.py"))
kewords_module = importlib.util.module_from_spec(spec)
sys.modules["kewords_app"] = kewords_module
spec.loader.exec_module(kewords_module)

app = kewords_module.app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
