from langgraph.graph import StateGraph,START,END
from utils.schema import State
from agents.universal_naming_agent import universal_name_refiner_v4
from agents.universal_parser_agent import parser_graph
from agents.comparison_agent import get_comparison_agent
from graph.process_chunk_func import remove_chunks, push_chunks
from graph.routing_after_parsing import routing_after_parsing
from graph.routing_to_parsing import routing_to_parsing

def main_graph():
    builder = StateGraph(State)

    builder.add_node("naming_agent",universal_name_refiner_v4)
    builder.add_node("parsing_agent",parser_graph)
    builder.add_node("archive_existing_chunks",remove_chunks)
    builder.add_node("push_all",push_chunks)
    builder.add_node("compare",get_comparison_agent)

    builder.add_edge(START,"naming_agent")

    builder.add_conditional_edges(
        "naming_agent",
        routing_to_parsing,
        {
            "parsing_agent":"parsing_agent",
            "end":END
        }
    )
    builder.add_conditional_edges(
        "parsing_agent",
        routing_after_parsing,
        {
            "archive_existing_chunks":"archive_existing_chunks",
            "push_all":"push_all",
            "compare":"compare",
            "end":END
        }
    )

    builder.add_edge("archive_existing_chunks","push_all")
    builder.add_edge("compare",END)
    builder.add_edge("push_all",END)

    graph = builder.compile()
    return graph