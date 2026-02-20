from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def list_models():
    models = client.models.list()
    youngest_models = sorted(models.data, key=lambda x: x.created)

    print(f"{len(models.data)} models available.")
    print("50 youngest models:")
    print()

    for model in youngest_models[-50:]:
        print(f"* {model.id}, {model.created}, {model.owned_by}")
    print()
    
    print("---")
        

if __name__ == "__main__":
    list_models()
