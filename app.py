import json
from typing import List

from pydantic import BaseModel

from alpaka.api.schema import Room, StructureInput, asend, awatch_room, watch_room
from alpaka.funcs import achat, apull, chat, pull
from arkitekt_next import easy, progress, register
from rath.operation import GraphQLException
from rekuest_next import acall, afind
from rekuest_next.api.schema import Node, aretrieveall
from rekuest_next.postmans.errors import PostmanException
from rekuest_next.utils import acall_raw


class Tools(BaseModel):
    tools: List[Node]

    def to_str(self):
        return "\n".join([f"{i.id}: {i.name} - {i.description}" for i in self.tools])


model = "deepseek-r1:1.5b"


@register
async def talk(room: Room):
    """
    Talk to Llama3.1

    Args:
        z_steps (int): The number of z steps to acquire.

    Returns:
        Image: The latest image.

    """

    print("Starting talk")

    await apull("deepseek-r1:1.5b")

    tools = Tools(tools=await aretrieveall())

    messages = [
        {"role": "system", "content": tools.to_str()},
        {
            "role": "system",
            "content": """
                
                You are an AI assitant that has the upper tools (so called nodes) at your disposable, they are listet as "ID: Name - Short descriptoin"
                
                You can to multiple tasks:
                    - YOu can perform the action that the user requested by replying
                      "{"use": $node_id, "args": $ARGS}"  be aware that args needs to be
                      a dictionary with the correct keys and values for the node (documented when searching for the node).
                      NEVER use a node if you are not 90% sure what it does.
                    - You can search for a node by replying "{"search": $node_id}". 
                      which will give oyu more information about the node
                    - You can saw a simple answer by replying "{"answer": "text"}"
                      if you require more information from the user.
                
                
                You can only reply with a json object, and nothing else. DO NOT REPLY with anything else than a json object.
                """,
        },
    ]

    async for i in awatch_room(room=room, agent_id="jhnnsrs"):
        messages.append({"role": "user", "content": i.message.text})

        answer = await achat(
            model=model,
            messages=messages,
        )

        train_of_thought = []

        iterations = 0

        while answer:
            iterations += 1
            print("Messages", messages)
            try:
                messages.append(
                    {"role": "system", "content": answer["message"]["content"]}
                )
                print(answer["message"]["content"])
                answer = answer["message"]["content"].strip()
                to_json = json.loads(answer)

                if "search" in to_json:
                    print("Searching")
                    active_node = await afind(to_json["search"])
                    messages.append(
                        {
                            "role": "user",
                            "content": "Here is the information about the node {}. Remember your instructions and if you need input do only reply in valid json as outlined before.".format(
                                active_node
                            ),
                        }
                    )
                    answer = await achat(
                        model=model,
                        messages=messages,
                    )
                    continue

                if "answer" in to_json:
                    await asend(room=room, text=to_json["answer"], agent_id="jhnnsrs")
                    answer = False
                    continue

                if "use" in to_json:
                    "Print using"
                    node = await afind(to_json["use"])

                    if node:
                        node.validate_args(**to_json["args"])

                    await asend(
                        room=room,
                        text=f"Yeah sure, using this node...",
                        agent_id="jhnnsrs",
                        attach_structures=[
                            StructureInput(
                                object=node.id, identifier="@rekuest-next/node"
                            )
                        ],
                    )
                    print("Calling node")
                    answer = await acall_raw(node=node.id, kwargs={**to_json["args"]})
                    messages.append(
                        {
                            "role": "user",
                            "content": "Here is the information about the call of the node {}. Remember your instructions.".format(
                                active_node
                            ),
                        }
                    )
                    await asend(
                        room=room,
                        text=f"Here is the return of the node",
                        agent_id="jhnnsrs",
                        attach_structures=[
                            StructureInput(
                                object=answer[port.key], identifier=port.identifier
                            )
                            for port in node.returns
                            if port.identifier
                        ],
                    )
                    answer = False

                else:
                    await asend(
                        room=room,
                        text="I am sorry, I do not understand",
                        agent_id="jhnnsrs",
                    )
                    answer = False

            except GraphQLException as e:
                messages.append(
                    {
                        "role": "user",
                        "content": f"You made an error in your query. Please try again: the error was {e}",
                    }
                )
                if not iterations > 3:
                    answer = await achat(
                        model=model,
                        messages=messages,
                    )
                else:
                    answer = False

            except PostmanException as e:
                print("PostmanError", e)

                if not iterations > 3:
                    answer = await achat(
                        model=model,
                        messages=messages,
                    )
                else:
                    answer = False
            except json.JSONDecodeError as e:
                print("Key error", e)

                messages.append(
                    {
                        "role": "user",
                        "content": """
                                 You did not reply with valid json. Please think again.
                                            
                                            You can to multiple tasks:
                                - YOu can perform the action that the user requested by replying
                                "{"use": $node_id, "args": $ARGS}"  be aware that args needs to be
                                a dictionary with the correct keys and values for the node (documented when searching for the node).
                                NEVER use a node if you are not 90% sure what it does.
                                - You can search for a node by replying "{"search": $node_id}". 
                                which will give oyu more information about the node
                                - You can saw a simple answer by replying "{"answer": "text"}"
                                if you require more information from the user.
                            
                            
                            You can only reply with a json object, and nothing else. DO NOT REPLY with anything else than a json object.
                                 
                                 """,
                    }
                )
                if not iterations > 3:
                    answer = await achat(
                        model=model,
                        messages=messages,
                    )
                else:
                    answer = False

            except ValueError as e:
                print("ValueError", e)

                messages.append(
                    {
                        "role": "user",
                        "content": """You didn't pass the correct arguments for the node. Please search first about the onde and then try again or ask the user for more information by promprint for an answer {"answer": "text you want to ask"}""",
                    }
                )
                if not iterations > 3:
                    answer = await achat(
                        model=model,
                        messages=messages,
                    )
                else:
                    answer = False

