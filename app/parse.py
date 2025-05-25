import re
from typing import Literal, Union

from SNP4 import *


def parse_position(d: dict) -> Position:
    x = d["x"]
    y = d["y"]

    return Position(x, y)


def parse_rule(s: str, from_xml: bool = False) -> Rule:
    if from_xml:
        result = re.match(r"(.*)/(\d*a)->(\d*a|0);(\d+)", s)

        if result:
            regex, consumed, produced, delay = result.groups()

            regex = Rule.xml_to_python_regex(regex)
            consumed = Rule.get_value(consumed, in_xml=True)
            produced = Rule.get_value(produced, in_xml=True)
            delay = int(delay)

            return Rule(regex, consumed, produced, delay)
        else:
            raise NameError("Error\tParsing xml rule failed...")
    else:
        result = re.match(r"^(.+)\s*/\s*(.+)\s*\\to\s+(.+)\s*;\s*(\d+)$", s)
        result_no_regex = re.match(r"^(.+)\s*\\to\s+(.+)\s*;\s*(\d+)$", s)
        result_no_delay = re.match(r"^(.+)\s*/(.+)\s*\\to\s*(.+)$", s)
        result_both = re.match(r"^(.+)\s*\\to\s*(.+)\s*$", s)

        if result is not None:
            regex_, consumed_, produced_, delay_ = result.groups()

            regex = Rule.json_to_python_regex(regex_)
            consumed = Rule.get_value(consumed_, in_xml=False)
            produced = Rule.get_value(produced_, in_xml=False)
            delay = int(delay_)

            return Rule(regex, consumed, produced, delay)

        elif result_no_regex is not None:
            consumed_, produced_, delay_ = result_no_regex.groups()

            regex = Rule.json_to_python_regex(consumed_)
            consumed = Rule.get_value(consumed_, in_xml=False)
            produced = Rule.get_value(produced_, in_xml=False)
            delay = int(delay_)

            return Rule(regex, consumed, produced, delay)

        elif result_no_delay is not None:
            regex_, consumed_, produced_ = result_no_delay.groups()

            regex = Rule.json_to_python_regex(regex_)
            consumed = Rule.get_value(consumed_, in_xml=False)
            produced = Rule.get_value(produced_, in_xml=False)
            delay = 0

            assert produced == 0

            return Rule(regex, consumed, produced, delay)

        elif result_both is not None:
            consumed_, produced_ = result_both.groups()

            regex = Rule.json_to_python_regex(consumed_)
            consumed = Rule.get_value(consumed_, in_xml=False)
            produced = Rule.get_value(produced_, in_xml=False)
            delay = 0

            assert produced == 0

            return Rule(regex, consumed, produced, delay)

        else:
            raise NameError("Error\tParsing non-xml rule failed...")


def parse_neuron(d: dict, from_xml: bool = False) -> Neuron:
    if from_xml:
        id = d["id"]
        type_: Literal["regular", "input", "output"] = (
            "input"
            if "isInput" in d and d["isInput"]
            else "output" if "isOutput" in d and d["isOutput"] else "regular"
        )
        position = Position(
            round(float(d["position"]["x"])), round(float(d["position"]["y"]))
        )
        rules = (
            list(map(lambda s: parse_rule(s, from_xml), d["rules"].split()))
            if "rules" in d
            else []
        )
        content: Union[int, list[int]] = (
            int(d["spikes"])
            if type_ == "regular"
            else (
                list(map(int, d["bitstring"].split(",")))
                if d["bitstring"] is not None
                else []
            )
        )
    else:
        id = d["id"]
        type_ = d["type"]
        position = parse_position(d["position"])
        rules = [parse_rule(rule) for rule in d["rules"]] if "rules" in d else []

        content: Union[int, list[int]] = (
            int(d["content"])
            if type_ == "regular"
            else list(map(int, list(d["content"]))) if len(d["content"]) > 0 else []
        )

    return Neuron(id, type_, position, rules, content)


def parse_synapse(d: dict) -> Synapse:
    from_ = d["from"]
    to = d["to"]
    weight = d["weight"]

    return Synapse(from_, to, weight)


def parse_dict(d: dict, from_xml: bool = False) -> System:
    if from_xml:
        to_index = {}
        for i, k in enumerate(d.keys()):
            to_index[k] = i

        neurons = [parse_neuron(v, from_xml) for v in d.values()]

        synapses = []
        for v in d.values():
            if "outWeights" in v:
                for k_, v_ in v["outWeights"].items():
                    synapses.append(Synapse(v["id"], k_, int(v_)))
    else:
        neurons = [parse_neuron(neuron, from_xml) for neuron in d["neurons"]]
        synapses = [parse_synapse(synapse) for synapse in d["synapses"]]

    return System(neurons, synapses)
