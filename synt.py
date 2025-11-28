from time import time
import os

import re
import json
from collections import defaultdict
import networkx as nx
from senstore.segmenter import segment_text
from config import CF
from chatbot import ask
from redir import redirect_edges_no_backflow


from vis import visualize_rels
from natlog.prolog_parser import parse_prolog_clause, parse_goal, VarNum


def fact_prompter(quest: str) -> str:
    return f"""
I am counting on and fully trusting your very high intelligence and linguistic skill. 
Here is our intellectual exercise: 
    Each time I will ask a question, your answers will be all just a list of 
    Subject,Verb,Object triplets,
    expressed as Prolog fact of the form: 
    
    fact(subject,verb,object). 
    
    No additional text or explanation, just the facts themselves, please!

    Here is my question: "{quest}"

    Please use "_" instead of spaces or camel-code in multi-word phrases!
    Please put all constants between single quotes, there should be 
    no capitalized variables in the facts!
    Please make sure your answers are syntactically correct Prolog terms!
""".strip()


def query_prompter(quest: str, facts: str) -> str:
    return f"""
I am counting on and fully trusting your very high intelligence and linguistic skill. 
Here is our intellectual exercise: 
You will convert the content of a question into a set of one or more Prolog goals of the form:

    fact(subject1,verb1,object1),fact(subject2,verb2,object2),...
     
    When a question requires ans answer in its subject or object part, you will use a variable, e.g., X or Y.
    For instance, a question like "Who is the president of the USA?" will be converted to:  

    fact(X, is, president_of_usa).

    For instance, a question like "Who discovered alternative current and used it in a motor?" will be converted to:

    fact(X, discovered, alternative_current), fact(X, used_alternative_current, in_a_motor).

    I am attaching also a set of Prolog facts tto which, if possible, the goals should be unified with.

    {facts}

    Finally, here is my question you will need to work with: "{quest}"

    No additional text or explanation, just the goals themselves, please!
    Please use "_" instead of spaces or camel-code in multi-word phrases!
    Please make sure your answers are syntactically correct Prolog terms!
""".strip()


def sum_prompter(facts: str) -> str:
    return f"""
I am counting on and fully trusting your very high intelligence and reasoning skills. 
Here is our intellectual exercise: 
I will send you a set of Prolog facts, each of the form:

    fact(Subject,Verb,Object). 

    Please summarize them in a few plain sentences,
    without any additional text or explanation, just the summary itself!

Here are the facts:

{facts} 
""".strip()


def next_quest_prompter(sum: str, quest0) -> str:
    return f"""
The topic I have started thinking about is {quest0}.

Here is a summary about the thoughts I am interested to explore in depth:

{sum}   

What short, salient question should I ask about it? Please make sure it is a genuine
follow-up question, that is not a rephrasing of the previous one! Also, try to make
the question focus on a single topic, not a conjunction of multiple questions!

Also, to be clear, I want to focus on the contents of the summary, not on meta-questions about it!
Please return just the question, without any additional text or explanation!
""".strip()


def gen_prompter(nouns: str, context: str) -> str:
    return f"""
I will send you a set of Prolog nouns separated by semicolons (;)
that you will need to generalize by creating S,V,O triplets
like 

   (noun, is_a_kind_of, more_general_concept) or 
   (noun, is_a_part_of, container_concept) or
   (noun, is_an_analog_of, well_known_concept) or
   (noun, is_a_consequence_of, known_cause) or
   (noun, can_lead_to, possible_consequence) or
   (noun, is_a_reason_for, well_known_effect)
   and so on.
   
You will work with the following context in mind, describing what
theee nouns are about:

{context}

Here are the nouns:

{nouns} 

you will work with.

Please return a list of Prolog facts of the form:
   
   fact(subject,verb,object)
 
for each such triplet you can think of.

Please use "_" instead of spaces or camel-code in multi-word phrases!
Please put all constants between single quotes, there should be no capitalized variables in the facts!
Please make sure your answers are syntactically correct Prolog terms!
""".strip()


def get_llm_name():
    if CF.USE_OLLAMA:
        return "ollama"
    else:
        return "gpt"


def camel_to_snake(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[\s\-]+", "_", s)  # spaces/dashes -> underscore
    s = re.sub(
        r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s
    )  # split XMLHTTPRequest -> XML_HTTPRequest
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)  # split fooBar -> foo_Bar
    s = re.sub(r"([A-Za-z])([0-9])", r"\1_\2", s)  # letters->digits boundary
    s = re.sub(r"([0-9])([A-Za-z])", r"\1_\2", s)  # digits->letters boundary
    return s.lower()


def onto_name(out_dir: str, quest: str) -> str:
    """Create a file name based on the question."""
    return (
        "out/"
        + get_llm_name()
        + "_"
        + "_".join(quest.lower().split()).replace('"', "").replace("?", "")[:60]
    )


def store_kb(ddict: defaultdict, out_dir: str, quest: str):
    """Store the knowledge base as a JSON file."""
    os.makedirs(out_dir, exist_ok=True)
    fname = onto_name(out_dir, quest)
    jname = fname + ".json"

    with open(jname, "w") as f:
        json.dump(ddict, f, indent=2)
    print(f"Knowledge graph stored in {jname}")


def step_with(prompter, *args) -> tuple[str, float]:
    """Create a prompt with the prompter and ask the LLM, returning the answer and cost."""
    prompt = prompter(*args)
    answer, cost = ask(prompt)
    return answer, cost


def uniform_str(s: str) -> str:
    """Convert a string to a uniform format for comparison."""
    s = s.strip().replace(" ", "_").replace("-", "_").replace("'", "")
    s = s.replace("+", "_and_").replace("/", "_or_")
    s = s.replace('"', "").replace("`", "")
    s = camel_to_snake(s)
    return s


def to_edge(f: str) -> tuple[str, str, str] | None:
    """Convert a Prolog fact string to an edge tuple, or return None if malformed."""
    if not f:
        return None
    try:
        clause = parse_prolog_clause(f)
        if not clause:
            print("WARNING: ignoring unexpected clause:", f, "-->", clause)
            return None
        edge = clause[0][0][1:]  # (pred, (s,v,o))
        if len(edge) != 3:
            print("WARNING: tuple of length 3 expected:", f, "-->", edge)
            return None

        for x in edge:
            if isinstance(x, VarNum):
                print("WARNING: ignoring fact with variable:", f, "-->", edge)
                return None

        # return edge
        s, v, o = edge
        if not good_noun(s) or not good_noun(o):
            print("WARNING: ignoring fact with bad noun:", f, "-->", edge)
            return None
        return (uniform_str(s), uniform_str(v), uniform_str(o))
    except Exception as e:
        print("WARNING: ignoring unparsable fact:", f, "Error:", e)
        return None


def onto_step(
    quest0: str, quest: str, ddict: defaultdict, edges: set
) -> tuple[str, float]:
    """Perform one step of the ontology building process."""
    t1 = time()
    facts, c1 = step_with(fact_prompter, quest)
    sum, c2 = step_with(sum_prompter, facts)
    new_quest, c3 = step_with(next_quest_prompter, sum, quest0)
    goal, c4 = step_with(query_prompter, quest, facts)

    facts = facts.split("\n")
    goal = goal.strip().replace('"', "").split("\n")

    clean_facts = []
    for f in facts:
        edge = to_edge(f)
        if edge is not None:
            edges.add(edge)
            clean_facts.append(f)
    facts = clean_facts

    sum = segment_text(sum)

    print("\nQUESTION:\n", quest)
    print("\nQUEST AS A GOAL:\n", goal)
    print("\nFACTS:\n", facts)
    print("\nSUMMARY:\n", sum)
    print("\nNEXT QUESTION:\n", new_quest)

    ddict[quest].append(
        {
            "QUERY_GOAL": goal,
            "ANSWER_FACTS": facts,
            "ANSWER_SUMMARY": sum,
            "NEXT_QUESTION": new_quest,
        }
    )

    cost = c1 + c2 + c3 + c4

    print("\nEDGES:")
    for svo in edges:
        print("\t len=", len(svo), svo)

    print("\nCost: $%.8f" % cost, "time:", time() - t1)

    return new_quest, cost


def good_noun(s: str) -> bool:
    """Check if a string is a good noun for generalization."""

    if not s:
        return False
    s = s.strip()
    if len(s) < 3:
        return False
    if len(s) > 42:
        return False
    if s.lower() in {
        "this",
        "that",
        "these",
        "those",
        "you",
        "we",
        "she",
        "they",
        "them",
        "what",
        "which",
        "who",
        "whom",
        "where",
        "when",
        "why",
        "how",
    }:
        return False
    return True


def gen_step(edges, context: str) -> tuple[set, float]:
    """Generate generalizations for a set of nouns in a given context."""

    nouns = ";".join(
        set(s for (s, _, _) in edges if isinstance(s, str) and good_noun(s))
    )

    gens, cost = step_with(gen_prompter, nouns, context)
    gens = gens.split("\n")
    gens = [to_edge(f) for f in gens]
    gens = set(f for f in gens if f is not None)

    print("\nGENERALIZATION EDGES:")
    for g in gens:
        print("\t len=", len(g), g)

    print("\nEDGES SO FAR:", len(edges))
    print("\nGENERALIZATIONS:", len(gens))
    return gens, cost


def onto_loop(quest0: str, n: int = 4, out_dir=None) -> tuple[str, defaultdict, float]:
    """Run the ontology building loop for n steps starting from quest0."""
    if out_dir is None:
        out_dir=CF.OUTDIR
    CF.show()
    quest = quest0
    ddict = defaultdict(list)
    edges = set()
    total_cost = 0
    t1 = time()
    for i in range(n):
        print(f"\n\n=== STEP {i+1} ===")
        quest, cost = onto_step(
            quest0, quest, ddict, edges
        )  # quest to edges + goal !!!!
        total_cost += cost
        store_kb(ddict, out_dir, quest0)  # could be moved outside the loop

    gens, c5 = gen_step(edges, quest0)  # from nouns to generalizations !!!!
    edges = edges | gens

    print("\nTOTAL EDGES:", len(edges))

    fname = onto_name(out_dir, quest0)

    save_files(
        fname, quest0, ddict, edges
    )  # Save the knowledge base, summary, and Prolog facts !!!

    edges = rank_svos(edges, CF.TOPN)  # Rank and filter the edges to keep the top N !!!
    _, vname = visualize_rels(edges, fname + "_graph", show=True)  # Visualize  !!!

    print(f"\nKnowledge graph shown in {vname}")

    CF.show()

    total_cost += c5
    print("\nTOTAL Cost: $%.8f" % total_cost, "total time:", time() - t1)
    return quest, ddict, total_cost


def save_files(fname: str, quest0: str, ddict: dict, edges: set):

    tname = fname + "_kb.tsv"
    with open(tname, "w") as f:
        for s, v, o in edges:
            f.write(f"{s}\t{v}\t{o}\n")
    print(f"Knowledge graph stored in {tname}")

    sname = fname + "_sum.txt"
    with open(sname, "w") as f:
        f.write(f"SEED QUESTION: {quest0}\n\n")
        for q, items in ddict.items():
            for item in items:

                for sent in item["ANSWER_SUMMARY"]:
                    f.write(sent + " ")
                f.write("\n\n")

            quest = item["NEXT_QUESTION"]
            f.write(f"QUESTION: {q}\n\n")

    print(f"Knowledge summary stored in {sname}")

    pname = fname + ".pro"
    with open(pname, "w") as f:
        f.write(f"% SEED QUESTION: {quest0}\n\n")
        for s, v, o in edges:
            f.write(f"fact({s},{v},{o}).\n")

    print(f"Knowledge facts stored in Prolog file {pname}")


def rank_svos(svos, topn, redirect=None):
    if redirect is None:
        redirect=CF.REDIRECT
    d = defaultdict(set)
    for s, v, o in svos:
        d[(s, o)].add(v)
    maxlen = max(len(vs) for vs in d.values())
    print(f"Max verbs per (s,o): {maxlen}")

    g = nx.DiGraph()
    for (s, o), vs in d.items():
        weight = maxlen / len(vs)  # smaller is better
        g.add_edge(s, o, weight=weight)
    rs = nx.pagerank(g.reverse())
    ranked = sorted(
        svos, key=lambda x: rs.get(x[0], 0) + rs.get(x[2], 0), reverse=True
    )  # REVERSE IS BETTER!!!
    if topn <= 0:
        return ranked

    if redirect:
        print(f"Redirecting to top {topn} edges based on PageRank")
        res = redirect_edges_no_backflow(ranked, rs, topn)
        return list(res)

    return ranked[0:topn]
