from collections import defaultdict
from synt import onto_loop, onto_step
from config import CF


def test0():
    onto_loop(
        "How can an FPGA be used to accelerate execution of a small instruction set?"
    )


def test1():
    onto_loop(
        "How can an FPAA be used to accelerate execution of a small instruction set?"
    )


def test2():
    onto_loop(
        "How I can remove noisy relations from a set of triplets extracted from a text document?",
    )


def test3():
    onto_loop(
        "How do ASP implementations avoid the high cost of computing stable models?"
    )


def test4():
    onto_loop(
        "What are  the reasons why Negation as Failure can be considered harmful in Logic Programming?",
    )


def test5():
    onto_loop(
        "How can we infer new logic facts from LLM generated S,V,O triplets?",
    )


def test6():
    onto_loop(
        "How can a small scale, quickly trainable transformer system be used to discover new AI architectures?",
    )


def test7():
    onto_loop(
        "How to extend a Horn Clause program to query a vector store of embeddings and abduce new clauses based on the matches?",
    )


def test8():
    onto_loop(
        "What are the advantages of requesting LLMs to answer in the form of Prolog facts?",
    )


def test9():
    onto_loop(
        "What are the disadvantages of requesting LLMs to answer in the form of Prolog facts?",
    )


def test10():
    onto_loop(
        "What can save knowledge graphs from deprecation in the age of Generative AI?",
    )


def test11():
    onto_loop(
        "How can the SpaceForce mitigate hypersonic missile threats?",
    )


def test12():
    onto_loop(
        "What are the merits of interpreting Horn Clause logic intuitionistically?",
    )


def test13():
    onto_loop(
        "What changes in the unification algorithm are needed to use fact embeddings in a vector store?",
    )


def test14():
    onto_loop(
        "What academic research fields are deprecated by the advent of Generative AI?",
    )


def test15():
    onto_loop(
        "Which symbolic AI academic research fields will benefit from the advent of Generative AI?",
    )


def test16():
    onto_loop(
        "What professions are at risk to be replaced by AI?",
    )


def test17():
    onto_loop(
        "What Symbolic AI fields are likely to become irrelevant with the advent of Generative AI?",
    )


def test18():
    onto_loop(
        "What Symbolic AI fields are likely to stay relevant with the advent of Generative AI?",
    )


def test19():
    onto_loop(
        "What kind of internal logic (classical, intuitionistic, non-monotonic, paraconsitent) best describes a reasoning LLM's output?"
    )


if __name__ == "__main__":
    CF.USE_OLLAMA = False  # type: ignore
    CF.GPT_MODEL = "gpt-5"
    test19()
