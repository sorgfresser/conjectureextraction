from openai import OpenAI
from anthropic import Client
from diophantineequations.prompts import INFORMAL_PROOF
import weave

# client = OpenAI()
client = Client()

@weave.op()
def sketch_proof(conjecture_formalized, conjecture_nl, parent_conjecture):
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[
            {
                "role": "user",
                "content": INFORMAL_PROOF + "\n\n" + "Natural language larger theorem\n" + parent_conjecture + "\n\n" + "Natural language conjecture\n" + conjecture_nl + "\n\nFormalized conjecture\n" + conjecture_formalized
            }
        ],
        max_tokens=1024
    )
    output = "\n".join([block.text for block in response.content])
    # response = client.chat.completions.create(
    #     model="o3-mini",
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": INFORMAL_PROOF + "\n\n" + "Natural language larger theorem\n" + parent_conjecture + "\n\n" + "Natural language conjecture\n" + conjecture_nl + "\n\nFormalized conjecture\n" + conjecture_formalized
    #         }
    #     ]
    # )
    # output = response.choices[0].message.content
    return output
