"""
The journal is the core datastructure in AIDE that contains:
- the generated code samples
- information how code samples relate to each other (the tree structure)
- code execution results
- evaluation information such as metrics
...
"""

import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional

from dataclasses_json import DataClassJsonMixin
from interpreter.interpreter_parallel import ExecutionResult
from utils.metric import MetricValue
from utils.response import trim_long_string
from search.node import Node

@dataclass
class Journal(DataClassJsonMixin):
    """A collection of nodes representing the solution tree."""

    nodes: list[Node] = field(default_factory=list)
    # eda: InteractiveSession = field(default_factory=lambda: InteractiveSession())

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    def __len__(self) -> int:
        """Return the number of nodes in the journal."""
        return len(self.nodes)

    def append(self, node: Node) -> None:
        """Append a new node to the journal."""
        node.step = len(self.nodes)
        self.nodes.append(node)

    @property
    def draft_nodes(self) -> list[Node]:
        """Return a list of nodes representing intial coding drafts"""
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> list[Node]:
        """Return a list of nodes that are considered buggy by the agent."""
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> list[Node]:
        """Return a list of nodes that are not considered buggy by the agent."""
        return [n for n in self.nodes if not n.is_buggy]

    def get_metric_history(self) -> list[MetricValue]:
        """Return a list of all metric values in the journal."""
        return [n.metric for n in self.nodes]

    def get_best_node(self, only_good=True) -> None | Node:
        """Return the best solution found so far (node with the highest validation metric)."""
        if only_good:
            nodes = self.good_nodes
            if not nodes:
                return None
        else:
            nodes = self.nodes
        return max(nodes, key=lambda n: n.metric)

    def generate_summary(self, include_code: bool = False) -> str:
        """Generate a summary of the journal for the agent."""
        summary = []
        for n in self.good_nodes:
            summary_part = f"Design: {n.plan}\n"
            if include_code:
                summary_part += f"Code: {n.code}\n"
            summary_part += f"Results: {n.analysis}\n"
            summary_part += f"Validation Metric: {n.metric.value}\n"
            summary.append(summary_part)
        return "\n-------------------------------\n".join(summary)
    
    def generate_summary_from_node(self, target_node : Node, include_code : bool = False) -> str:
        """Generate a summary, based on the parent/target node"""
        summary = []
        history_nodes = []
        related_nodes = []

        current_node = target_node
        while current_node.parent:
            current_node = current_node.parent
            history_nodes.append(current_node)
        
        if target_node.parent:
            related_nodes = [n for n in target_node.parent.children if n != target_node][:5]
        
        history_nodes = history_nodes + related_nodes

        for n in history_nodes:
            summary_part = f"Design: {n.plan}\n"
            if include_code:
                summary_part += f"Code: {n.code}\n"
            summary_part += f"Results: {n.analysis}\n"
            summary_part += f"Validation Metric: {n.metric.value}\n"
            summary.append(summary_part)
        return "\n-------------------------------\n".join(summary)

def get_path_to_node(journal: Journal, node_id: str) -> list[str]:
    path = [node_id]

    node2parent = {n.id: n.parent.id for n in journal.nodes if n.parent is not None}
    while node_id in node2parent:
        parent_id = node2parent[node_id]
        path.append(parent_id)
        node_id = parent_id
    return path[::-1]


def get_longest_path(journal: Journal) -> list[str]:
    longest_path = []
    for node in journal.nodes:
        path = get_path_to_node(journal, node.id)
        if len(path) > len(longest_path):
            longest_path = path
    return longest_path


def filter_on_path(journal: Journal, path: list[str]) -> Journal:
    journal_copy = copy.deepcopy(journal)
    journal_copy.nodes = [n for n in journal.nodes if n.id in path]
    # further filter nodes, setting their _term_out and exc_stack to "<OMITTED>"
    for n in journal_copy.nodes:
        n._term_out = "<OMITTED>"
        n.exc_stack = "<OMITTED>"

    return journal_copy


def filter_for_best_path(journal: Journal, best_node: str) -> Journal:
    path_to_best = get_path_to_node(journal, best_node)
    filtered_journal = filter_on_path(journal, path_to_best)
    return filtered_journal


def filter_for_longest_path(journal: Journal) -> Journal:
    longest_path = get_longest_path(journal)
    filtered_journal = filter_on_path(journal, longest_path)
    return filtered_journal


def filter_journal(journal: Journal) -> Journal:
    best_node = journal.get_best_node(only_good=True)

    if best_node is not None:
        filtered_journal = filter_for_best_path(journal, best_node.id)
    else:
        filtered_journal = filter_for_longest_path(journal)

    return filtered_journal