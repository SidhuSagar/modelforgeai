import os

# Define the project structure
project_structure = {
    "ModelForgeAI": {
        "core": [
            "dataset_handler.py",
            "prompt_parser.py",
            "model_builder.py",
            "trainer.py",
            "evaluator.py",
            "utils.py"
        ],
        "models": [
            "__init__.py",
            "base_model.py"
        ],
        "data": {
            "raw": [],
            "processed": [],
            "splits": []
        },
        "outputs": {
            "logs": [],
            "reports": [],
            "generated_code": []
        },
        "configs": [
            "default.yaml",
            "experiment1.yaml"
        ],
        "tests": [
            "test_phase1.py",
            "test_phase2.py",
            "test_utils.py"
        ],
        "scripts": [
            "run_training.sh",
            "prepare_data.py"
        ],
        "docs": [
            "architecture.md",
            "usage.md"
        ],
        "root_files": [
            "main.py",
            "README.md",
            "requirements.txt",
            "setup.py",
            ".gitignore"
        ]
    }
}

def create_structure(base_path, structure):
    """ Recursively create folders and files """
    for key, value in structure.items():
        project_path = os.path.join(base_path, key)
        os.makedirs(project_path, exist_ok=True)

        for sub_key, sub_value in value.items() if isinstance(value, dict) else []:
            create_structure(project_path, {sub_key: sub_value})

        # If value is a list, create files
        if isinstance(value, list):
            for file in value:
                file_path = os.path.join(project_path, file)
                with open(file_path, "w") as f:
                    f.write("")  # Create empty file

# Run the script
base_directory = r"C:\Users\Dell\Desktop"  # Change if needed
create_structure(base_directory, project_structure)

print("✅ Project structure created successfully at:", base_directory)
