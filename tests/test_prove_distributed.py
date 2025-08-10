from diophantineequations.lemma_prove import prove_distributed, ProofSample
from diophantineequations.distributed_models import WorkerResponse, TacticsResponse
from lean_repl_py import LeanREPLHandler
from typing import Any


class MockComm:
    def __init__(self, communication_stack: list[list[WorkerResponse]]):
        self.communication_stack = communication_stack
        self.worker_index = [0] * len(communication_stack)
        self.message_counts = [0] * len(communication_stack)

    def Get_size(self):
        return len(self.communication_stack) + 1  # +1 for master

    def Iprobe(self, source):
        source -= 1  # 1-indexed
        return self.message_counts[source] > 0

    def recv(self, source) -> dict[str, Any]:
        source -= 1  # 1-indexed
        message = self.communication_stack[source][self.worker_index[source]]
        self.worker_index[source] += 1
        self.message_counts[source] -= 1
        return message.model_dump()

    def send(self, message: dict[str, Any], dest):
        self.message_counts[dest - 1] += 1

    def all_finished(self):
        return all(
            [self.worker_index[i] == len(self.communication_stack[i]) for i in range(len(self.communication_stack))])


def test_prove_distributed():
    theorem = "theorem and_pq (p q : Prop) (a : p) (b : q) : p ∧ q := by sorry"

    handler = LeanREPLHandler()
    handler.send_command(theorem)
    response, env = handler.receive_json()
    proof_state = response["sorries"][0]
    comm = MockComm([[
        WorkerResponse(response=TacticsResponse(tactics=["constructor", "somethingfail"],
                                                goal="p q : Prop\na : p\nb : q\n⊢ p ∧ q", tactic_scores=[-1.0, -3.0])),
        WorkerResponse(response=TacticsResponse(tactics=["somethingfail", "exact a"],
                                                goal='case left\np q : Prop\na : p\nb : q\n⊢ p', tactic_scores=[-2.0, -0.5])),
        WorkerResponse(
            response=TacticsResponse(tactics=["exact b"], goal='case right\np q : Prop\na : p\nb : q\n⊢ q', tactic_scores=[-1.0])), ]])

    solved, proof, _ = prove_distributed(handler, proof_state, [], comm)
    assert solved
    assert proof == "constructor\nexact a\nexact b"
    assert comm.all_finished()


def test_prove_distributed_multiple_workers():
    theorem = "theorem and_pq (p q : Prop) (a : p) (b : q) : p ∧ q := by sorry"

    handler = LeanREPLHandler()
    handler.send_command(theorem)
    response, env = handler.receive_json()
    proof_state = response["sorries"][0]
    # We have two workers, but the second one is never used because exact a and assumption lead to the same goal
    comm = MockComm([[
        WorkerResponse(response=TacticsResponse(tactics=["constructor", "somethingfail"],
                                                goal="p q : Prop\na : p\nb : q\n⊢ p ∧ q", tactic_scores=[1.0, 0.0])),
        WorkerResponse(response=TacticsResponse(tactics=["somethingfail", "exact a", "assumption"],
                                                goal='case left\np q : Prop\na : p\nb : q\n⊢ p', tactic_scores=[0.0, 1.0])),
        WorkerResponse(
            response=TacticsResponse(tactics=["exact b"], goal='case right\np q : Prop\na : p\nb : q\n⊢ q', tactic_scores=[1.0])), ],
        []])

    solved, proof, _ = prove_distributed(handler, proof_state, [], comm)
    assert solved
    assert proof == "constructor\nexact a\nexact b"
    assert comm.all_finished()


def test_get_proof():
    theorem = "theorem and_pq (p q : Prop) (a : p) (b : q) : p ∧ q := by sorry"

    handler = LeanREPLHandler()
    handler.send_command(theorem)
    response, env = handler.receive_json()
    proof_state = response["sorries"][0]
    # We have two workers, but the second one is never used because exact a and assumption lead to the same goal
    comm = MockComm([[
        WorkerResponse(response=TacticsResponse(tactics=["constructor", "somethingfail"],
                                                goal="p q : Prop\na : p\nb : q\n⊢ p ∧ q", tactic_scores=[1.0, 0.0])),
        WorkerResponse(response=TacticsResponse(tactics=["somethingfail", "exact a", "assumption"],
                                                goal='case left\np q : Prop\na : p\nb : q\n⊢ p', tactic_scores=[0.0, 1.0])),
        WorkerResponse(
            response=TacticsResponse(tactics=["exact b"], goal='case right\np q : Prop\na : p\nb : q\n⊢ q', tactic_scores=[0.0])), ],
        []])

    solved, proof, proofs = prove_distributed(handler, proof_state, [], comm, get_proofs=True)
    assert solved
    assert proof == "constructor\nexact a\nexact b"
    assert comm.all_finished()
    assert proofs == [
        ProofSample(goal='p q : Prop\na : p\nb : q\n⊢ p ∧ q', premises=[], proof='constructor\nexact a\nexact b',
                    all_goals=['p q : Prop\na : p\nb : q\n⊢ p ∧ q'], tactic='constructor'),
        ProofSample(goal='case left\np q : Prop\na : p\nb : q\n⊢ p', premises=[], proof='exact a\nexact b',
                    all_goals=['case left\np q : Prop\na : p\nb : q\n⊢ p', 'case right\np q : Prop\na : p\nb : q\n⊢ q'],
                    tactic='exact a'),
        ProofSample(goal='case right\np q : Prop\na : p\nb : q\n⊢ q', premises=[], proof='exact b',
                    all_goals=['case right\np q : Prop\na : p\nb : q\n⊢ q'], tactic='exact b')]
