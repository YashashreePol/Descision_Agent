from typing import Dict, TypedDict, List, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END

# --------------------------------------------------
# STATE
# --------------------------------------------------
class DecisionWorkflowState(TypedDict):
    messages: List[BaseMessage]
    workflow_params: Dict[str, Any]

    is_complete: bool
    needs_user_input: bool
    input_request: Optional[str]
    pending_workflow_step: Optional[str]
    last_node: str  # for debugging


# --------------------------------------------------
# DETERMINISTIC MESSAGE GENERATOR (NO LLM)
# --------------------------------------------------
def generate_message(context_key: str) -> str:
    """Deterministic, template-based message generator."""
    templates = {
        "start": "Hello! I'm your support assistant. How can I help you today?",
        "unclear": "I didn't quite understand. Could you please describe your issue in more detail?",
        "billing_missing_invoice": "For billing issues, please provide your invoice ID or transaction reference.",
        "technical_missing_error": "To diagnose technical issues, I need the exact error message or steps to reproduce.",
        "general_too_vague": "Could you specify what topic you'd like help with? (e.g., 'login', 'subscription', 'export data')",
        "fallback": "I need a bit more information to assist you properly.",
    }
    return templates.get(context_key, templates["fallback"])


# --------------------------------------------------
# CHAT NODE
# --------------------------------------------------
def chat_node(state: DecisionWorkflowState) -> DecisionWorkflowState:
    # On first run, greet user
    if not state["messages"]:
        msg = generate_message("start")
        return {
            **state,
            "messages": [AIMessage(content=msg)],
            "input_request": msg,
            "needs_user_input": True,
            "last_node": "chat",
        }

    # If we're here because a handler asked for more info, do nothing.
    # The message was already added by the handler.
    # We just signal that we're ready for input (handled externally).
    return {**state, "last_node": "chat"}


# --------------------------------------------------
# VALIDATION NODE
# --------------------------------------------------
def validation_node(state: DecisionWorkflowState) -> DecisionWorkflowState:
    text = state["workflow_params"].get("user_request", "").strip()

    if not text or len(text) < 5:
        msg = generate_message("unclear")
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=msg)],
            "needs_user_input": True,
            "input_request": msg,
            "pending_workflow_step": "validation_node",
            "last_node": "validation",
        }

    # Valid input: proceed
    state["workflow_params"]["user_request"] = text
    return {
        **state,
        "pending_workflow_step": None,
        "last_node": "validation",
    }


# --------------------------------------------------
# DECISION NODE (RULE-BASED)
# --------------------------------------------------
def decision_node(state: DecisionWorkflowState) -> DecisionWorkflowState:
    text = state["workflow_params"]["user_request"].lower()

    if any(k in text for k in ["refund", "payment", "invoice", "charged", "bill", "subscription"]):
        category = "BILLING"
    elif any(k in text for k in ["crash", "error", "bug", "not working", "fail", "404", "500", "freeze"]):
        category = "TECHNICAL"
    else:
        category = "GENERAL"

    state["workflow_params"]["category"] = category
    return {**state, "last_node": "decision"}


# --------------------------------------------------
# RESULT NODE (NEW - AS PER BOILERPLATE)
# --------------------------------------------------
def result_node(state: DecisionWorkflowState) -> DecisionWorkflowState:
    """Final node that generates the output message and ends the workflow."""
    final_response = state["workflow_params"].get("final_response", "No response was generated.")
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=final_response)],
        "is_complete": True,
        "last_node": "result_node",
    }


# --------------------------------------------------
# HANDLERS (NOW ONLY PREPARE DATA — NO MESSAGES APPENDED)
# --------------------------------------------------
def technical_handler_node(state: DecisionWorkflowState) -> DecisionWorkflowState:
    text = state["workflow_params"]["user_request"].lower()
    if not any(kw in text for kw in ["error", "crash", "bug", "exception", "failed", "not working"]):
        msg = generate_message("technical_missing_error")
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=msg)],
            "needs_user_input": True,
            "input_request": msg,
            "is_complete": False,
            "last_node": "technical_handler",
        }

    response = f"""
🔧 TECHNICAL SUPPORT SUMMARY

Overview:
Your request has been carefully reviewed and identified as a technical-related issue.

User Context:
You reported the following problem:
"{state['workflow_params']['user_request']}"

Diagnosis Insight:
Application crashes often occur due to software bugs, outdated versions,
corrupted cache, or compatibility issues with the system environment.

Recommended Actions:
1. Restart the application and try again.
2. Ensure you are using the latest version of the app.
3. Check for pending system updates.
4. If the issue persists, reinstall the application.
"""

    # Only store response — DO NOT append message
    state["workflow_params"]["final_response"] = response.strip()
    return {
        **state,
        "is_complete": True,
        "last_node": "technical_handler",
    }


def billing_handler_node(state: DecisionWorkflowState) -> DecisionWorkflowState:
    text = state["workflow_params"]["user_request"].lower()
    if "invoice" not in text and "inv-" not in text and "transaction" not in text and "refund" not in text:
        msg = generate_message("billing_missing_invoice")
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=msg)],
            "needs_user_input": True,
            "input_request": msg,
            "is_complete": False,
            "last_node": "billing_handler",
        }

    response = f"""
💳 BILLING SUPPORT SUMMARY

Overview:
Your request falls under billing-related concerns.

User Context:
"{state['workflow_params']['user_request']}"

Analysis:
Billing issues typically involve duplicate charges, failed payments,
or subscription discrepancies.

Next Steps:
We’ve logged your request. If a refund is applicable, it will be processed
within 5–7 business days.
"""

    # Only store response — DO NOT append message
    state["workflow_params"]["final_response"] = response.strip()
    return {
        **state,
        "is_complete": True,
        "last_node": "billing_handler",
    }


def general_handler_node(state: DecisionWorkflowState) -> DecisionWorkflowState:
    text = state["workflow_params"]["user_request"]
    words = text.split()
    if len(words) < 3 and "?" not in text:
        msg = generate_message("general_too_vague")
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=msg)],
            "needs_user_input": True,
            "input_request": msg,
            "is_complete": False,
            "last_node": "general_handler",
        }

    response = f"""
ℹ️ GENERAL SUPPORT RESPONSE

Overview:
Your request has been identified as a general inquiry.

Your Question:
"{text}"

Guidance:
This type of question is usually answered through product guides,
FAQs, or official documentation.

Next Steps:
Please refer to the help resources available on our platform
for detailed and up-to-date information.
"""

    # Only store response — DO NOT append message
    state["workflow_params"]["final_response"] = response.strip()
    return {
        **state,
        "is_complete": True,
        "last_node": "general_handler",
    }


# --------------------------------------------------
# ROUTERS
# --------------------------------------------------
def route_from_chat(state: DecisionWorkflowState):
    if state["pending_workflow_step"]:
        return state["pending_workflow_step"]
    return "validation_node"


def route_from_validation(state: DecisionWorkflowState):
    return "chat_node" if state["needs_user_input"] else "decision_node"


def route_from_decision(state: DecisionWorkflowState):
    cat = state["workflow_params"]["category"]
    if cat == "BILLING":
        return "billing_handler_node"
    if cat == "TECHNICAL":
        return "technical_handler_node"
    return "general_handler_node"


def route_from_handler(state: DecisionWorkflowState):
    if not state["is_complete"]:
        return "chat_node"
    return "result_node"  # ← CHANGED: now goes to result_node


# --------------------------------------------------
# BUILD GRAPH
# --------------------------------------------------
graph = StateGraph(DecisionWorkflowState)

graph.add_node("chat_node", chat_node)
graph.add_node("validation_node", validation_node)
graph.add_node("decision_node", decision_node)
graph.add_node("billing_handler_node", billing_handler_node)
graph.add_node("technical_handler_node", technical_handler_node)
graph.add_node("general_handler_node", general_handler_node)
graph.add_node("result_node", result_node)  # ← ADDED

graph.set_entry_point("chat_node")

graph.add_conditional_edges("chat_node", route_from_chat)
graph.add_conditional_edges("validation_node", route_from_validation)
graph.add_conditional_edges("decision_node", route_from_decision)
graph.add_conditional_edges("billing_handler_node", route_from_handler)
graph.add_conditional_edges("technical_handler_node", route_from_handler)
graph.add_conditional_edges("general_handler_node", route_from_handler)

graph.add_edge("result_node", END)  # ← ADDED

agent = graph.compile()


# --------------------------------------------------
# RUNNER — FIXED TO AVOID INFINITE LOOP
# --------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 Starting Decision Agent Workflow...\n")

    # Start with a clean state
    state: DecisionWorkflowState = {
        "messages": [],
        "workflow_params": {},
        "is_complete": False,
        "needs_user_input": False,
        "input_request": None,
        "pending_workflow_step": None,
        "last_node": "init",
    }

    while True:
        # If no messages yet, greet the user (outside graph)
        if not state["messages"]:
            greeting = generate_message("start")
            print(f"\n🤖 Assistant: {greeting}")
            state["messages"].append(AIMessage(content=greeting))

        # Always get user input before running graph
        user_input = input("\n👤 User: ").strip()
        if not user_input:
            continue

        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))
        state["workflow_params"]["user_request"] = user_input
        state["needs_user_input"] = False
        state["input_request"] = None
        state["pending_workflow_step"] = None
        state["is_complete"] = False

        # Run the graph ONE TIME with this input
        try:
            state = agent.invoke(state, {"recursion_limit": 50})
        except Exception as e:
            print(f"\n⚠️ Error during processing: {e}")
            # Optionally reset or break
            break

        # After graph runs, check if done
        if state["is_complete"]:
            break

    # Final output
    print("\n" + "=" * 60)
    print("✅ FINAL RESPONSE")
    print("=" * 60)
    for msg in state["messages"]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        print(f"\n[{role}]: {msg.content}")