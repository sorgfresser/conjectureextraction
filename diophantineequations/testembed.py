import torch
from FlagEmbedding import BGEM3FlagModel
import sqlmodel
from diophantineequations.distributed.workers.conjecturer import SQLConjecture


engine = sqlmodel.create_engine("sqlite:////home/simon/PycharmProjects/diophantineequations/nodeepseekkiminaformalizevalid")


def format_doc_only_f(decl: str) -> str:
    return f'''Formal Declaration: {decl[:1536]}'''

batch_size = 64
model_path = "purewhite42/dependency_retriever_f"

with sqlmodel.Session(engine) as session:
    data = session.exec(sqlmodel.select(SQLConjecture)).all()

# Filter out None values and format the formal problem statements
formal_statements = []
informal_statements = []
for sample in data:
    if sample.formal_problem is None:
        continue
    formal_statements.append(sample.formal_problem)
    informal_statements.append(sample.informal_problem)

formal_statements = [format_doc_only_f(p) for p in formal_statements]
informal_statements = [informal_statement for informal_statement in informal_statements]
assert len(formal_statements) == len(informal_statements), "Formal and informal statements should have the same length at this point."
model = BGEM3FlagModel(model_path, use_fp16=True)

formal_embeddings = model.encode(
    formal_statements,
    batch_size=batch_size,
    max_length=1024)['dense_vecs']
informal_embeddings = model.encode(
    informal_statements,
    batch_size=batch_size,
    max_length=1024
)['dense_vecs']

similarity = (torch.tensor(informal_embeddings).double() @ torch.tensor(formal_embeddings).double().T)

diagonals = similarity.diagonal()

mean_diagonal = diagonals.mean()
print(f"Mean diagonal similarity: {mean_diagonal.item()}")