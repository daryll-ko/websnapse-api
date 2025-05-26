from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Optional, Protocol, Union

# import xmltodict
import yaml


class FileFormat(Protocol):
    def str_to_dict(self, s: str) -> dict: ...
    def dict_to_str(self, d: dict) -> str: ...


# class XML:
#     def str_to_dict(self, s: str) -> dict:
#         return xmltodict.parse(s)["content"]
# 
#     def dict_to_str(self, d: dict) -> str:
#         return xmltodict.unparse(d, pretty=True, newl="\n", indent=" " * 4)


class JSON:
    def str_to_dict(self, s: str) -> dict:
        return json.loads(s)

    def dict_to_str(self, d: dict) -> str:
        return json.dumps(d, indent=2)


class YAML:
    def str_to_dict(self, s: str) -> dict:
        return yaml.load(s, Loader=yaml.Loader)

    def dict_to_str(self, d: dict) -> str:
        return yaml.dump(d, sort_keys=False, indent=2)


def str_to_format(s: str) -> FileFormat:
    if s == ".xml":
        raise Exception(":(")
    elif s == ".json":
        return JSON()
    elif s == ".yaml":
        return YAML()
    else:
        raise Exception(":(")
    # match s:
    #     case ".xml":
    #         return XML()
    #     case ".json":
    #         return JSON()
    #     case ".yaml":
    #         return YAML()
    #     case _:
    #         raise Exception(":(")


@dataclass
class Rule:
    regex: str
    consumed: int
    produced: int
    delay: int

    def stringify(self, in_xml: bool) -> str:
        regex_ = (
            Rule.python_to_xml_regex(self.regex)
            if in_xml
            else Rule.python_to_json_regex(self.regex)
        )
        consumed_ = Rule.get_symbol(self.consumed, in_xml)
        to_ = "->" if in_xml else "\\to " if self.produced > 0 else "\\to"
        produced_ = Rule.get_symbol(self.produced, in_xml)
        head_ = (
            f"{regex_}/{consumed_}" if in_xml or regex_ != consumed_ else f"{regex_}"
        )
        delay_ = f";{self.delay}"
        return f"{head_}{to_}{produced_}{delay_ if in_xml or self.produced > 0 else ''}"

    @staticmethod
    def get_value(symbol: str, in_xml: bool) -> int:
        if in_xml:
            if symbol == "0":
                return 0
            elif symbol == "a":
                return 1
            else:
                return int(symbol.replace("a", ""))
        else:
            if symbol == "\\lambda" or symbol == "0":
                return 0
            elif symbol == "a":
                return 1
            else:
                result = re.match(r"a\^\{?(\d+)\}?", symbol)
                return int(result.groups()[0]) if result is not None else -1

    @staticmethod
    def get_symbol(value: int, in_xml: bool) -> str:
        if value == 0:
            return "0" if in_xml else "\\lambda"
        elif value == 1:
            return "a"
        else:
            return f"{value}a" if in_xml else f"a^{{{value}}}"

    @staticmethod
    def json_to_python_regex(s: str) -> str:
        substituted = re.sub(
            r"\\cup",
            "|",
            re.sub(
                r"\^\{?\+\}?",
                "+",
                re.sub(
                    r"\^\{?\*\}?",
                    "*",
                    re.sub(r"\^\{?(\d+)\}?", r"{\1}", s),
                ),
            ),
        ).replace(" ", "")
        return f"^{substituted}$"

    @staticmethod
    def python_to_json_regex(s: str) -> str:
        return re.sub(
            r"\|",
            r" \\cup ",
            re.sub(
                r"\+",
                r"^{+}",
                re.sub(r"\*", r"^{*}", re.sub(r"\{(\d+)\}", r"^{\1}", s[1:-1])),
            ),
        )

    @staticmethod
    def xml_to_python_regex(s: str) -> str:
        substituted = re.sub(r"(\d+)a", r"a{\1}", s)
        return f"^{substituted}$"

    @staticmethod
    def python_to_xml_regex(s: str) -> str:
        return re.sub(r"a\{(\d+)\}", r"\1a", s[1:-1])


@dataclass
class Position:
    x: int
    y: int

    def to_dict(self) -> dict[str, Any]:
        return vars(self)


@dataclass
class Synapse:
    from_: str
    to: str
    weight: int

    def to_dict(self) -> dict[str, Any]:
        return {"from": self.from_, "to": self.to, "weight": self.weight}


@dataclass
class Neuron:
    id: str
    type_: Literal["regular", "input", "output"]
    position: Position
    rules: list[Rule]
    content: Union[int, list[int]]

    def add(self, val: int) -> None:
        if self.type_ == "regular":
            assert isinstance(self.content, int)
            self.content += val
        elif self.type_ == "output":
            assert isinstance(self.content, list)
            self.content.append(val)
        else:
            assert False
        # match self.type_:
        #     case "regular":
        #         assert isinstance(self.content, int)
        #         self.content += val
        #     case "output":
        #         assert isinstance(self.content, list)
        #         self.content.append(val)
        #     case "input":
        #         assert False

    def remove(self, val: int) -> None:
        if self.type_ == "regular":
            assert isinstance(self.content, int)
            self.content -= val
        else:
            assert False
        # match self.type_:
        #     case "regular":
        #         assert isinstance(self.content, int)
        #         self.content -= val
        #     case _:
        #         assert False

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "type": self.type_,
            "position": self.position.to_dict(),
            "content": (
                self.content
                if isinstance(self.content, int)
                else "".join(map(str, self.content))
            ),
        }
        if self.type_ == "regular":
            d["rules"] = [rule.stringify(in_xml=False) for rule in self.rules]
        return d


class System:
    neurons: list[Neuron]
    synapses: list[Synapse]

    _id_to_neuron: dict[str, Neuron]
    _neuron_to_index: dict[str, int]
    _adjacency_list: list[list[Synapse]]
    _incoming_spikes: dict[int, dict[str, int]]
    _downtime: list[int]
    _buffers: list[Optional[Rule]]

    def __init__(self, neurons: list[Neuron], synapses: list[Synapse]) -> None:
        self.neurons = neurons
        self.synapses = synapses

        self._id_to_neuron = {}
        for neuron in self.neurons:
            self._id_to_neuron[neuron.id] = neuron

        self._neuron_to_index = {}
        for i, neuron in enumerate(self.neurons):
            self._neuron_to_index[neuron.id] = i

        self._adjacency_list = [[] for _ in range(len(self.neurons))]
        for i, neuron in enumerate(self.neurons):
            self._adjacency_list[i] = self._get_synapses_from(neuron.id)

        self._incoming_spikes = defaultdict(lambda: defaultdict(int))

        self._downtime = [0 for _ in range(len(self.neurons))]

        self._buffers = [None for _ in range(len(self.neurons))]

    def __repr__(self) -> str:
        lines = []
        cols = max(len(neuron.id) for neuron in self.neurons)
        for i, neuron in enumerate(self.neurons):
            lines.append(
                f"{neuron.id:<{cols+5}}{neuron.content}{'/' + str(self._downtime[i]) if neuron.type_ == 'regular' else ''}"
            )
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "neurons": [neuron.to_dict() for neuron in self.neurons],
            "synapses": [synapse.to_dict() for synapse in self.synapses],
        }

    def _get_synapses_from(self, from_: str) -> list[Synapse]:
        return list(filter(lambda synapse: synapse.from_ == from_, self.synapses))

    def _get_synapses_to(self, to: str) -> list[Synapse]:
        return list(filter(lambda synapse: synapse.to == to, self.synapses))

    @staticmethod
    def clean_xml_tag(s: str) -> str:
        cleaned = re.sub(",", "", re.sub("}", "", re.sub("{", "", s)))
        if re.match(r"^\d+", s):
            return f"n_{cleaned}"
        else:
            return f"{cleaned}"

    def to_dict_xml(self) -> dict[str, Any]:
        neuron_entries: list[tuple[str, dict[str, Any]]] = []

        for neuron in self.neurons:
            k = System.clean_xml_tag(neuron.id)
            v: dict[str, Any] = {
                "id": System.clean_xml_tag(neuron.id),
                "position": {
                    "x": neuron.position.x,
                    "y": neuron.position.y,
                },
            }

            if len(neuron.rules) > 0:
                v["rules"] = " ".join(
                    list(map(lambda rule: rule.stringify(in_xml=True), neuron.rules))
                )

            if isinstance(neuron.content, int):
                v["spikes"] = neuron.content
                v["delay"] = 0
            else:
                v["bitstring"] = (
                    ",".join(map(str, neuron.content))
                    if neuron.content is not None
                    else ""
                )

            if neuron.type_ == "input":
                assert isinstance(neuron.content, list)
                v["delay"] = 0
                v["isInput"] = True

            if neuron.type_ == "output":
                v["spikes"] = 0
                v["isOutput"] = True

            for synapse in self.synapses:
                if synapse.from_ == neuron.id:
                    if "out" not in v:
                        v["out"] = []
                    if "outWeights" not in v:
                        v["outWeights"] = {}
                    v["out"].append(System.clean_xml_tag(synapse.to))
                    v["outWeights"][System.clean_xml_tag(synapse.to)] = synapse.weight

            neuron_entries.append((k, v))

        return {"content": dict(neuron_entries)}

    def _get_possible_rules(self) -> dict[str, list[Rule]]:
        ans = {}

        for i, neuron in enumerate(self.neurons):
            if neuron.type_ == "regular":
                assert isinstance(neuron.content, int)

                if self._downtime[i] == 0:
                    possible_rules = []

                    for rule in neuron.rules:
                        result = re.match(rule.regex, "a" * neuron.content)
                        if result:
                            possible_rules.append(rule)

                    if possible_rules:
                        ans[neuron.id] = possible_rules
                elif self._downtime[i] == 1:
                    self._downtime[i] = 0

                    rule = self._buffers[i]

                    assert rule is not None

                    rule_copy = deepcopy(rule)
                    rule_copy.delay = 0

                    ans[neuron.id] = [rule_copy]

                    self._buffers[i] = None
                else:
                    self._downtime[i] -= 1

        return ans

    @staticmethod
    def _choose_possible_rules(
        l: dict[str, list[Rule]], acc: dict[str, Rule] = {}
    ) -> Iterator[dict[str, Rule]]:
        if len(l) == 0:
            yield acc
        else:
            for k, v in l.items():
                for rule in v:
                    l_copy = deepcopy(l)
                    acc_copy = deepcopy(acc)

                    l_copy.pop(k)
                    acc_copy[k] = rule

                    yield from System._choose_possible_rules(l_copy, acc_copy)
                break

    def _apply_chosen_rules(self, l: dict[str, Rule] = {}) -> None:
        affected = set()

        for id, rule in l.items():
            neuron = self._id_to_neuron[id]
            i = self._neuron_to_index[neuron.id]
            if neuron.type_ == "regular":
                if rule.delay == 0:
                    neuron.remove(rule.consumed)
                    if rule.produced > 0:
                        for synapse in self._adjacency_list[i]:
                            to_neuron = self.neurons[self._neuron_to_index[synapse.to]]
                            to_neuron.add(rule.produced * synapse.weight)
                            affected.add(to_neuron.id)
                else:
                    self._buffers[i] = rule
                    self._downtime[i] = rule.delay

        for neuron in self.neurons:
            if neuron.type_ == "output" and neuron.id not in affected:
                neuron.add(0)

    def get_spike_distance(self) -> Optional[int]:
        for neuron in self.neurons:
            if neuron.type_ == "output":
                assert isinstance(neuron.content, list)
                l = -1
                for i, b in enumerate(neuron.content):
                    if b > 0:
                        if l == -1:
                            l = i
                        else:
                            return i - l
                break

    def get_bit_string(self) -> Optional[str]:
        for neuron in self.neurons:
            if neuron.type_ == "output":
                assert isinstance(neuron.content, list)
                return "".join(map(str, neuron.content))

    def is_done(self) -> bool:
        return self.get_spike_distance() is not None

    def get_next_det(self) -> System:
        return next(self.get_next_nondet())

    def _get_inputs(self):
        for neuron in self.neurons:
            if neuron.type_ == "input":
                assert isinstance(neuron.content, list)
                if len(neuron.content) > 0 and neuron.content[0] == 1:
                    for synapse in self._adjacency_list[
                        self._neuron_to_index[neuron.id]
                    ]:
                        to, weight = synapse.to, synapse.weight
                        to_neuron = self.neurons[self._neuron_to_index[to]]
                        to_neuron.add(weight)
                neuron.content = neuron.content[1:]

    def get_next_nondet(self) -> Iterator[System]:
        possible_rules = self._get_possible_rules()
        self._get_inputs()

        if len(possible_rules) > 0:
            for chosen_rules in self._choose_possible_rules(possible_rules):
                clone = deepcopy(self)
                clone._apply_chosen_rules(chosen_rules)
                yield clone
        else:
            clone = deepcopy(self)
            clone._apply_chosen_rules()
            yield clone

    def accepts_dis(self, n: int) -> bool:
        for ans in self.get_configs(n + 10, det=False, lazy=True):
            if ans.get_spike_distance() == n:
                return True
        return False

    def accepts_lit(self, b: str) -> bool:
        for ans in self.get_configs(len(b), det=False, lazy=True):
            if ans.get_bit_string() == b:
                return True
        return False

    def get_configs(self, time_left: int, det: bool, lazy: bool) -> Iterator[System]:
        # iterative

        st = []
        st.append((0, deepcopy(self)))
        res = []

        while st:
            t, sys = st.pop()
            res.append(sys)
            if t < time_left and (not lazy or not sys.is_done()):
                op = sys.get_next_det if det else sys.get_next_nondet
                for poss in list(op()):
                    st.append((t + 1, poss))

        yield from res

        # recursive

        # if time_left == 0 or (lazy and self.is_done()):
        #     yield self
        # else:
        #     if det:
        #         yield from self.get_next_det().get_configs(time_left - 1, det, lazy)
        #     else:
        #         for poss in self.get_next_nondet():
        #             yield from poss.get_configs(time_left - 1, det, lazy)

    def simulate(
        self,
        type_: Literal["generating", "halting", "boolean"],
        time_limit: int,
        make_log: bool,
    ) -> int:
        simulation_log: list[str] = []
        print_buffer: list[str] = []

        def flush_print_buffer() -> None:
            simulation_log.append("\n".join(print_buffer))
            print_buffer.clear()

        def capture_state() -> None:
            for i, neuron in enumerate(self.neurons):
                if neuron.type_ == "regular":
                    print_buffer.append(
                        f">> {neuron.id}\t<{neuron.content}/{self._downtime[i]}>"
                    )
                else:
                    print_buffer.append(f">> {neuron.id}\t{neuron.content}")

        start, end = -1, -1
        boolean_result = -1
        t = 0
        done = False

        for i, neuron in enumerate(self.neurons):
            if neuron.type_ == "input":
                assert isinstance(neuron.content, list)
                for t, spikes in enumerate(neuron.content):
                    if spikes > 0:
                        for synapse in self._adjacency_list[
                            self._neuron_to_index[neuron.id]
                        ]:
                            to, weight = synapse.to, synapse.weight
                            j = self._neuron_to_index[to]
                            self._incoming_spikes[t][to] += spikes

        while not done and t < time_limit:

            # = = = = = = = = = = = = = = = = = = = = = = = = =

            simulation_log.append(f"{t=}")
            simulation_log.append("> phase 1: incoming spikes")

            incoming_updates: Counter[str] = Counter()

            for neuron in self.neurons:
                i = self._neuron_to_index[neuron.id]
                if self._downtime[i] == 0:
                    if neuron.id in self._incoming_spikes[t]:
                        incoming_updates[neuron.id] += self._incoming_spikes[t][
                            neuron.id
                        ]
                        if neuron.type_ == "regular":
                            assert isinstance(neuron.content, int)
                            neuron.content += incoming_updates[neuron.id]

            for k, v in incoming_updates.items():
                print_buffer.append(f">> {k}\t{v}")

            if len(print_buffer) > 0:
                flush_print_buffer()
            else:
                simulation_log.append(">> no events during phase 1")

            # = = = = = = = = = = = = = = = = = = = = = = = = =

            simulation_log.append("> phase 2: showing starting state")

            capture_state()
            flush_print_buffer()

            # = = = = = = = = = = = = = = = = = = = = = = = = =

            simulation_log.append("> phase 3: selecting rules")

            some_rule_selected = False

            for i, neuron in enumerate(self.neurons):
                if neuron.type_ == "regular":
                    assert isinstance(neuron.content, int)

                    if self._downtime[i] == 0:
                        possible_rules = []

                        for j, rule in enumerate(neuron.rules):
                            result = re.match(rule.regex, "a" * neuron.content)
                            if result:
                                possible_rules.append(rule)

                        if len(possible_rules) > 0:
                            some_rule_selected = True
                            rule = random.choice(possible_rules)
                            print_buffer.append(f">> {neuron.id}: {rule}")

                            neuron.content -= rule.consumed
                            if rule.produced > 0:
                                for synapse in self._adjacency_list[
                                    self._neuron_to_index[neuron.id]
                                ]:
                                    to, weight = synapse.to, synapse.weight
                                    j = self._neuron_to_index[to]
                                    self._incoming_spikes[
                                        t
                                        + rule.delay
                                        + (
                                            1
                                            if self.neurons[j].type_ != "output"
                                            else 0
                                        )
                                    ][to] += (rule.produced * weight)

                            self._downtime[i] = rule.delay
                    else:
                        self._downtime[i] -= 1

            if len(print_buffer) > 0:
                flush_print_buffer()
            else:
                simulation_log.append(">> no events during phase 3")

            done = (
                all(len(d) == 0 for _t, d in self._incoming_spikes.items() if _t > t)
                and not some_rule_selected
            )

            # = = = = = = = = = = = = = = = = = = = = = = = = =

            simulation_log.append("> phase 4: accumulating updates, detecting outputs")

            output_detected = False

            for i, neuron in enumerate(self.neurons):
                if self._downtime[i] == 0:
                    if neuron.id in self._incoming_spikes[t]:
                        delta = self._incoming_spikes[t][neuron.id]
                        incoming_updates[neuron.id] += delta

                    if neuron.type_ == "output":
                        assert isinstance(neuron.content, list)
                        neuron.content.append(incoming_updates[neuron.id])
                        output_detected |= incoming_updates[neuron.id] > 0

            if len(print_buffer) > 0:
                flush_print_buffer()
            else:
                simulation_log.append(">> no events during phase 4")

            if type_ == "generating" and output_detected:
                if start == -1:
                    start = t
                    simulation_log.append(">> detected first output spike")
                else:
                    end = t
                    simulation_log.append(
                        ">> detected second output spike, wrapping up..."
                    )
                    break

            # = = = = = = = = = = = = = = = = = = = = = = = = =

            simulation_log.append("> phase 5: showing in-between state")

            capture_state()
            flush_print_buffer()

            if type_ == "boolean" and t == 3:
                boolean_result = output_detected
                break

            t += 1

        if make_log:
            for line in simulation_log:
                print(line)
                print()

        if type_ == "generating":
            if end == -1:
                return -1
            else:
                return end - start
        elif type_ == "halting":
            return t
        elif type_ == "boolean":
            return boolean_result

        # match type_:
        #     case "generating":
        #         if end == -1:
        #             return -1
        #         else:
        #             return end - start
        #     case "halting":
        #         return t
        #     case "boolean":
        #         return boolean_result

    def simulate_using_matrices(self):
        to_index = {}
        for j, neuron in enumerate(self.neurons):
            to_index[neuron.id] = j

        N = sum([len(neuron.rules) for neuron in self.neurons])
        M = len(self.neurons)

        P = [[0 for _ in range(M)] for _ in range(N)]  # production matrix (N×M)
        C = [[0 for _ in range(M)] for _ in range(N)]  # consumption matrix (N×M)

        offset = 0
        for j, neuron in enumerate(self.neurons):
            adjacent_indices = [
                to_index[synapse.to] for synapse in self._get_synapses_from(neuron.id)
            ]
            for i, rule in enumerate(neuron.rules):
                for adjacent_index in adjacent_indices:
                    P[offset + i][adjacent_index] = rule.produced
                C[offset + i][j] = rule.consumed
            offset += len(neuron.rules)

        time = 0

        while time < 10**3:
            # S = [0 for _ in range(M)]  # status vector (1×M)
            # I = [0 for _ in range(N)]  # indicator vector (1×N)

            # SP  # spiking vector (1×N)
            # G = I • P  # gain vector (1×N • N×M = 1×M)
            # L = SP • C  # loss vector (1×N • N×M = 1×M)

            # NG = S × (G - L) (1×M)
            # C_{k+1} - C_{k} = S × (G - L)
            # C_{k+1} = C_{k} + S × [(I • P) - (Sp • C)]

            # what's the difference between I and SP?

            time += 1

        raise NotImplementedError()
