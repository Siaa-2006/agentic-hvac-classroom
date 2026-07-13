import os
import sys
import importlib

def check_files():
    """Verifies that all core codebase files exist in the directory."""
    required_files = ["app.py", "translator.py", "hvac_agents.py", "dataset.json"]
    print("🔍 [Phase 1] Checking Workspace Files...")
    
    all_exist = True
    for f in required_files:
        if os.path.exists(f):
            print(f"  ✅ {f} found.")
        else:
            print(f"  ❌ {f} is MISSING from this folder!")
            all_exist = False
    return all_exist

def test_imports():
    """Tests if all required python libraries are installed locally."""
    libraries = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "pydantic": "pydantic",
        "pandas": "pandas",
        "langgraph": "langgraph",
        "langchain_core": "langchain-core",
        "langchain_groq": "langchain-groq"
    }
    
    print("\n📦 [Phase 2] Checking Local Python Library Imports...")
    missing_libs = []
    
    for lib_name, pip_name in libraries.items():
        try:
            importlib.import_module(lib_name)
            print(f"  ✅ {lib_name} imported successfully.")
        except ImportError:
            print(f"  ❌ {lib_name} (Pip package: '{pip_name}') is not installed.")
            missing_libs.append(pip_name)
            
    return missing_libs

def test_api_keys():
    """Checks if the GROQ_API_KEY is configured in the environment."""
    print("\n🔑 [Phase 3] Checking Environment Configurations...")
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        # Mask key for safety
        masked_key = groq_key[:8] + "..." + groq_key[-4:] if len(groq_key) > 12 else "Present"
        print(f"  ✅ GROQ_API_KEY detected local variable: {masked_key}")
        return True
    else:
        print("  ⚠️  GROQ_API_KEY is not set in your local environment variables.")
        print("     (You can still deploy to Render if you inject it there, but local runs will fail.)")
        return False

def generate_requirements():
    """Generates a clean, locked requirements.txt file tailored for Render deployment."""
    print("\n📄 [Phase 4] Generating Production requirements.txt for Render...")
    requirements_content = """fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.4
pandas==2.2.2
langgraph==0.0.60
langchain-core==0.2.9
langchain-groq==0.1.5
requests==2.32.3
"""
    try:
        with open("requirements.txt", "w") as f:
            f.write(requirements_content)
        print("  ✅ requirements.txt has been generated/overwritten successfully.")
    except Exception as e:
        print(f"  ❌ Failed to write requirements.txt: {str(e)}")

def main():
    print("=" * 60)
    print("        CLASSROOM AGENTIC HVAC - DAY 1 INITIALIZER")
    print("=" * 60)
    
    files_ok = check_files()
    missing_libs = test_imports()
    test_api_keys()
    generate_requirements()
    
    print("\n" + "=" * 60)
    print("📋 DAY 1 ACTION SUMMARY:")
    print("=" * 60)
    
    if not files_ok:
        print("❌ Action Required: Ensure your terminal directory contains app.py, translator.py, and hvac_agents.py before proceeding.")
        sys.exit(1)
        
    if missing_libs:
        print("❌ Action Required: Install missing packages locally by running:")
        print(f"   pip install {' '.join(missing_libs)}")
        sys.exit(1)
        
    print("🚀 Your workspace is configured properly! You are ready to deploy to Render.")
    print("   Proceed to Git setup, push to GitHub, and deploy your new web service.")
    print("=" * 60)

if __name__ == "__main__":
    main()
