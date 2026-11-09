"""Read-only GraphQL for the model: search a backend's schema, describe a type, run a query.

Nothing here knows which backend it talks to: an engine is a name, a sentence about what
lives there, the service's own rath (the app's token, so the model sees what the app may
see), and the SDL its client package ships, as the fallback when introspection fails.
Changing something is what actions are for; mutations and subscriptions are refused.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from graphql import (
    GraphQLEnumType,
    GraphQLError,
    GraphQLInputObjectType,
    GraphQLInterfaceType,
    GraphQLObjectType,
    GraphQLSchema,
    GraphQLUnionType,
    OperationDefinitionNode,
    OperationType,
    Undefined,
    build_client_schema,
    build_schema,
    get_introspection_query,
    get_named_type,
    parse,
    validate,
)
from rath.operation import GraphQLException

logger = logging.getLogger(__name__)

GRAPHQL_SEARCH_TOOL = "graphql_search_schema"
GRAPHQL_DESCRIBE_TOOL = "graphql_describe"
GRAPHQL_QUERY_TOOL = "graphql_query"
GRAPHQL_TOOLS = (GRAPHQL_SEARCH_TOOL, GRAPHQL_DESCRIBE_TOOL, GRAPHQL_QUERY_TOOL)
#: Where the model goes instead, to change something.
ACTIONS_TOOL = "find_actions"
#: Characters of one tool result the model gets to see.
MAX_TOOL_CHARS = 6000


def clip(text: str, hint: str) -> str:
    """Cap a tool result, and say what to do about it rather than just cutting."""
    if len(text) <= MAX_TOOL_CHARS:
        return text
    return f"{text[:MAX_TOOL_CHARS]}\n... [truncated, {len(text)} characters in total. {hint}]"


def brief(text: str | None, limit: int = 160) -> str:
    """A description as one line: whitespace collapsed, first sentence-ish."""
    collapsed = " ".join((text or "").split())
    return collapsed if len(collapsed) <= limit else f"{collapsed[: limit - 3]}..."


def render_arguments(args: dict[str, Any]) -> str:
    if not args:
        return ""
    rendered = []
    for name, arg in args.items():
        default = ""
        if arg.default_value is not Undefined and arg.default_value is not None:
            default = f" = {json.dumps(arg.default_value, default=str)}"
        rendered.append(f"{name}: {arg.type}{default}")
    return f"({', '.join(rendered)})"


def render_field(name: str, field: Any) -> str:
    """One field as a line of SDL, with its description as a trailing comment."""
    line = f"{name}{render_arguments(getattr(field, 'args', None) or {})}: {field.type}"
    if getattr(field, "deprecation_reason", None):
        line += " @deprecated"
    if field.description:
        line += f"  # {brief(field.description)}"
    return line


def is_internal(name: str) -> bool:
    """Introspection (`__Type`) and federation (`_Entity`, `_service`) plumbing."""
    return name.startswith("_")


class GraphQLEngine:
    """One GraphQL service the model may read: a schema to explore, a client to ask.

    Nothing here knows which service it is talking to. ``get_rath`` returns the
    service's entered rath client -- the app's own, so queries run with the
    app's token and see what the app may see -- and everything else is derived
    from the schema.

    The schema comes from introspecting the live server, because that is the
    schema the query will actually be run against. ``bundled_schema`` (the SDL a
    client package ships for its codegen) is the fallback for servers that
    disable introspection or cap query depth below what the introspection query
    needs. The fallback is trusted less: a query that fails validation against
    introspected schema is refused locally, but one that fails against a bundled
    schema is sent anyway, since the file may simply be older than the server.

    Mutations are refused unless ``allow_mutations`` is set, and it is not set
    anywhere: this is how the model *looks*. Changing something is what actions
    are for -- they are typed, assignable and show up as tasks, none of which is
    true of a delete the model wrote by hand. Subscriptions are refused always;
    a tool call that never returns is not a tool call.
    """

    def __init__(
        self,
        name: str,
        description: str,
        get_rath: Callable[[], Any],
        bundled_schema: Path | None = None,
        allow_mutations: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.get_rath = get_rath
        self.bundled_schema = bundled_schema
        self.allow_mutations = allow_mutations
        #: "introspection" or "bundled", once the schema has been loaded.
        self.source: str | None = None
        self._schema: GraphQLSchema | None = None
        self._index: list[tuple[str, str, str, str]] | None = None

    # -- schema ------------------------------------------------------------

    async def schema(self) -> GraphQLSchema:
        """The service's schema, loaded once per process."""
        if self._schema is None:
            try:
                result = await self.get_rath().aquery(get_introspection_query(descriptions=True))
                self._schema = build_client_schema(result.data)
                self.source = "introspection"
            except Exception as error:
                if self.bundled_schema is None:
                    raise
                logger.warning(
                    "Could not introspect %s (%s); using the bundled schema.", self.name, error
                )
                # `assume_valid_sdl`: a dumped federation schema applies @key
                # without defining it, which is invalid SDL and a fine schema.
                self._schema = build_schema(
                    self.bundled_schema.read_text(), assume_valid_sdl=True
                )
                self.source = "bundled"
        return self._schema

    def roots(self, schema: GraphQLSchema) -> list[GraphQLObjectType]:
        """The root types the model is allowed to enter through."""
        roots = [schema.query_type]
        if self.allow_mutations:
            roots.append(schema.mutation_type)
        return [root for root in roots if root is not None]

    def index(self, schema: GraphQLSchema) -> list[tuple[str, str, str, str]]:
        """Everything searchable, as (kind, label, name, description).

        Three kinds, in the order they are useful: ``entry`` is a root field
        (somewhere a query can start), ``type`` is a named type, ``field`` is a
        field of one. Root types are left out of the latter two so that a root
        field is listed once, as an entry point.
        """
        if self._index is None:
            roots = self.roots(schema)
            entries: list[tuple[str, str, str, str]] = []
            for root in roots:
                for name, field in root.fields.items():
                    if not is_internal(name):
                        entries.append(("entry", f"{root.name}.{name}", name, field.description or ""))
            hidden = {schema.query_type, schema.mutation_type, schema.subscription_type}
            for name, named in schema.type_map.items():
                if is_internal(name) or named in hidden:
                    continue
                entries.append(("type", name, name, named.description or ""))
                for field_name, field in (getattr(named, "fields", None) or {}).items():
                    entries.append(
                        ("field", f"{name}.{field_name}", field_name, field.description or "")
                    )
            self._index = entries
        return self._index

    # -- the three things the model can do ---------------------------------

    async def search(self, query: str, limit: int = 15) -> str:
        """Entry points, types and fields matching ``query``; all entry points if empty."""
        schema = await self.schema()
        # Crudely singular, so that asking for "scenes" finds `scene` as well.
        tokens = {
            t[:-1] if len(t) > 3 and t.endswith("s") else t
            for t in re.split(r"[^a-z0-9_]+", query.lower())
            if len(t) > 2
        }

        if not tokens:
            # Without arguments: 54 signatures do not fit the cap, 54 names do,
            # and the arguments are one `describe` away.
            lines = [
                f"- {root.name}.{name}: {field.type}"
                + (f"  # {brief(field.description, 100)}" if field.description else "")
                for root in self.roots(schema)
                for name, field in root.fields.items()
                if not is_internal(name)
            ]
            return clip(
                f"{self.name}: {len(lines)} entry points. Describe one for its arguments, "
                f"or the type it returns for its fields.\n" + "\n".join(lines),
                "Search with a term instead.",
            )

        # Enough that a loosely matching entry point (`sceneSnapshots`) still
        # outranks an exactly matching field: there are thirteen `X.scene`
        # fields, and none of them is where a query can start.
        bonus = {"entry": 6, "type": 4, "field": 0}
        scored = []
        for kind, label, name, description in self.index(schema):
            lowered, text = name.lower(), description.lower()
            scores = [
                8
                if lowered in (token, f"{token}s")  # `scenes` is `scene`, exactly
                else 4
                if token in lowered
                else 1
                if token in text
                else 0
                for token in tokens
            ]
            matched = sum(1 for score in scores if score)
            if matched:
                # How much of the question an entry covers comes first -- for
                # "voxel size" a description mentioning both beats a field that
                # is merely called `size` -- then how well, then shorter names:
                # `scene` before `sceneLayerLinks`.
                scored.append(
                    (-matched, -(sum(scores) + bonus[kind]), len(label), kind, label, description)
                )

        if not scored:
            return f"Nothing in the {self.name} schema matches '{query}'. Search with an empty query to list every entry point."
        lines = [
            f"- [{kind}] {label}" + (f"  # {brief(description)}" if description else "")
            for *_, kind, label, description in sorted(scored)[: max(1, min(limit, 40))]
        ]
        return clip(
            f"{len(scored)} matches in {self.name}, best first:\n" + "\n".join(lines),
            "Narrow the search.",
        )

    async def describe(self, name: str) -> str:
        """A type, a `Type.field`, or a bare root field, rendered exactly."""
        schema = await self.schema()
        name = name.strip()

        owner_name, _, field_name = name.partition(".")
        if field_name:
            owner = schema.type_map.get(owner_name)
            field = (getattr(owner, "fields", None) or {}).get(field_name)
            if field is not None:
                return self._describe_field(owner_name, field_name, field)
        elif name in schema.type_map and not name.startswith("__"):
            return clip(self._describe_type(schema, name), "Describe a single field as Type.field.")
        else:
            for root in self.roots(schema):
                if name in root.fields:
                    return self._describe_field(root.name, name, root.fields[name])

        known = [entry[1] for entry in self.index(schema)] + [
            entry[2] for entry in self.index(schema) if entry[0] == "entry"
        ]
        close = difflib.get_close_matches(name, known, n=5, cutoff=0.6)
        suggestion = f" Did you mean: {', '.join(close)}?" if close else ""
        return f"'{name}' is not a type or field of the {self.name} schema.{suggestion}"

    def _describe_field(self, owner: str, name: str, field: Any) -> str:
        lines = [f"{owner}.{name}: {field.type}"]
        if field.description:
            lines.append(brief(field.description, 600))
        if getattr(field, "deprecation_reason", None):
            lines.append(f"Deprecated: {field.deprecation_reason}")
        args = getattr(field, "args", None) or {}
        if args:
            lines.append("Arguments:")
            lines.extend(f"  {render_field(arg_name, arg)}" for arg_name, arg in args.items())
        returned = get_named_type(field.type)
        if getattr(returned, "fields", None) or isinstance(returned, (GraphQLUnionType, GraphQLEnumType)):
            lines.append(f"Describe '{returned.name}' for what it holds.")
        return "\n".join(lines)

    def _describe_type(self, schema: GraphQLSchema, name: str) -> str:
        named = schema.type_map[name]
        lines: list[str] = []

        if isinstance(named, (GraphQLObjectType, GraphQLInterfaceType)):
            keyword = "type" if isinstance(named, GraphQLObjectType) else "interface"
            implements = ", ".join(i.name for i in named.interfaces)
            lines.append(f"{keyword} {name}" + (f" implements {implements}" if implements else ""))
            lines.extend(f"  {render_field(n, f)}" for n, f in named.fields.items())
            if isinstance(named, GraphQLInterfaceType):
                possible = ", ".join(t.name for t in schema.get_possible_types(named))
                lines.append(f"Implemented by: {possible}. Select their own fields with `... on Type`.")
        elif isinstance(named, GraphQLInputObjectType):
            lines.append(f"input {name}")
            lines.extend(f"  {render_field(n, f)}" for n, f in named.fields.items())
        elif isinstance(named, GraphQLEnumType):
            lines.append(f"enum {name}")
            lines.extend(
                f"  {value_name}" + (f"  # {brief(value.description)}" if value.description else "")
                for value_name, value in named.values.items()
            )
        elif isinstance(named, GraphQLUnionType):
            members = " | ".join(t.name for t in named.types)
            lines.append(f"union {name} = {members}. Select fields with `... on Type`.")
        else:
            lines.append(f"scalar {name}")

        if named.description:
            lines.insert(1, f"  # {brief(named.description, 600)}")

        # The way in: a type is only useful to the model once it knows which
        # query hands one out.
        entries = [
            f"{root.name}.{field_name}"
            for root in self.roots(schema)
            for field_name, field in root.fields.items()
            if get_named_type(field.type) is named
        ]
        if entries:
            lines.append(f"Returned by: {', '.join(entries)}")
        return "\n".join(lines)

    async def execute(self, query: str, variables: dict[str, Any] | None = None) -> str:
        """Run one read-only operation and return its data as JSON."""
        try:
            document = parse(query)
        except GraphQLError as error:
            return f"The query did not parse. {error.message}"

        operations = [d for d in document.definitions if isinstance(d, OperationDefinitionNode)]
        if len(operations) != 1:
            return f"Send exactly one operation per call; this document has {len(operations)}."
        kind = operations[0].operation
        if kind == OperationType.SUBSCRIPTION or (
            kind == OperationType.MUTATION and not self.allow_mutations
        ):
            return (
                f"Refused: {self.name} is read-only here, and this is a {kind.value}. "
                f"To change something, find an action with {ACTIONS_TOOL}."
            )

        problems = validate(await self.schema(), document)
        if problems and self.source == "introspection":
            listing = "\n".join(f"- {problem.message}" for problem in problems[:8])
            return (
                f"The query does not match the {self.name} schema and was not sent:\n{listing}\n"
                f"Use {GRAPHQL_DESCRIBE_TOOL} on the types involved."
            )

        try:
            result = await self.get_rath().aquery(query, variables or {})
        except GraphQLException as error:
            return f"The {self.name} server rejected the query: {error.message}"
        except Exception as error:  # noqa: BLE001 - any failure is text for the model
            logger.warning("GraphQL query against %s failed: %s", self.name, error)
            return f"The query against {self.name} failed: {error}"

        return clip(
            json.dumps(result.data, default=str, separators=(",", ":")),
            "Lower the pagination limit or select fewer fields.",
        )


def bundled_schema(client: Any) -> Path:
    """The SDL a client package ships for its codegen: ``<package>/api/schema.graphql``."""
    package = __import__(type(client).__module__.split(".")[0])
    return Path(package.__file__).parent / "api" / "schema.graphql"


#: The backends the model can read: name -> what lives there.
SERVICES = {
    "mikro": "microscopy data: images, scenes, layers, array datasets, tables, meshes, ROIs "
    "and annotations, and the coordinate systems that place them",
    "rekuest": "apps and what they do: agents, actions, implementations, tasks and their state",
    "alpaka": "rooms, messages and the language models",
    "kabinet": "the app store: apps, releases, definitions, and the pods they run in",
}


def engines_for(clients: dict[str, Any]) -> dict[str, GraphQLEngine]:
    """One engine per backend in ``SERVICES``, reached through the turn's own client.

    Built per turn, because the clients are the ones handed to this task; the schema is
    therefore loaded once per turn. Another backend is one more ``SERVICES`` entry.
    """
    return {
        name: GraphQLEngine(
            name=name,
            description=SERVICES[name],
            get_rath=lambda client=client: client.rath,
            bundled_schema=bundled_schema(client),
        )
        for name, client in clients.items()
        if name in SERVICES
    }


def graphql_tools(engines: dict[str, GraphQLEngine]) -> list[dict]:
    """The engine's three tools. One set for every service, told apart by `service`.

    A set per service would read a little better and scale a lot worse: tool
    schemas are sent on every round, and three more per backend adds up faster
    than one more enum value does.
    """
    if not engines:
        return []
    services = "; ".join(f"{e.name} ({e.description})" for e in engines.values())
    service = {
        "type": "string",
        "enum": list(engines),
        "description": f"Which backend. Available: {services}.",
    }

    def tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {"service": service, **properties},
                    "required": ["service", *required],
                },
            },
        }

    return [
        tool(
            GRAPHQL_SEARCH_TOOL,
            "Search a backend's GraphQL schema for entry points (root query fields), types "
            "and fields by name or description. An empty query lists every entry point.",
            {
                "query": {"type": "string", "description": "Keywords, e.g. 'scene layer'."},
                "limit": {"type": "integer", "description": "Max matches.", "default": 15},
            },
            [],
        ),
        tool(
            GRAPHQL_DESCRIBE_TOOL,
            "Show one schema element exactly: a type ('Scene') with all its fields, an input "
            "or enum with its members, or a field ('Query.scenes', 'Scene.layers', or just "
            "'scenes') with its arguments. Do this before writing a query.",
            {"name": {"type": "string", "description": "Type name, Type.field, or root field."}},
            ["name"],
        ),
        tool(
            GRAPHQL_QUERY_TOOL,
            "Run one read-only GraphQL query against a backend and get the data as JSON. "
            "Mutations are refused. Paginate lists and select only what you need; long "
            "results are truncated.",
            {
                "query": {"type": "string", "description": "The GraphQL document."},
                "variables": {"type": "object", "description": "Variables, if the query declares any."},
            },
            ["query"],
        ),
    ]


async def run_graphql_tool(
    name: str, arguments: dict[str, Any], engines: dict[str, GraphQLEngine]
) -> str:
    # Defaulted rather than demanded: with one backend there is nothing to choose.
    service = arguments.get("service") or next(iter(engines))
    engine = engines.get(service)
    if engine is None:
        return f"Unknown service '{service}'. Available: {', '.join(engines)}."

    try:
        if name == GRAPHQL_SEARCH_TOOL:
            return await engine.search(arguments.get("query") or "", arguments.get("limit") or 15)
        if name == GRAPHQL_DESCRIBE_TOOL:
            return await engine.describe(arguments.get("name") or "")
        variables = arguments.get("variables")
        if isinstance(variables, str):
            # Some models send the object as a JSON string.
            variables = json.loads(variables) if variables.strip() else None
        return await engine.execute(arguments.get("query") or "", variables)
    except Exception as error:  # noqa: BLE001 - any failure is text for the model
        # Same reasoning as a failing action: the model can act on this.
        logger.warning("%s on %s failed: %s", name, service, error)
        return f"{name} failed: {error}"
