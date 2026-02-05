"""Main controller for comfy-pilot.

Handles HTTP endpoints and coordinates agent communication.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional

from aiohttp import web

from .agents import AgentRegistry, AgentMessage, AgentConfig
from .agents.ollama import OllamaBackend  # noqa: F401 - registers itself
from .system import SystemMonitor
from .workflow import WorkflowManipulator


class ComfyPilotController:
    """Main controller handling HTTP routes and agent coordination."""

    def __init__(self):
        self.conversations: Dict[str, List[AgentMessage]] = {}

    def setup_routes(self, routes: web.RouteTableDef) -> None:
        """Register HTTP routes with aiohttp."""

        @routes.get("/comfy-pilot/agents")
        async def get_agents(request: web.Request) -> web.Response:
            """Get list of available agents."""
            agents = await AgentRegistry.get_available_agents()
            return web.json_response(agents)

        @routes.get("/comfy-pilot/system")
        async def get_system_info(request: web.Request) -> web.Response:
            """Get system information (GPU, models, etc.)."""
            gpu_info = await SystemMonitor.get_gpu_info()
            return web.json_response(gpu_info)

        @routes.get("/comfy-pilot/models")
        async def get_models(request: web.Request) -> web.Response:
            """Get available ComfyUI models."""
            models = await SystemMonitor.get_available_models()
            return web.json_response(models)

        @routes.get("/comfy-pilot/custom-nodes")
        async def get_custom_nodes(request: web.Request) -> web.Response:
            """Get installed custom nodes information."""
            custom_nodes = await SystemMonitor.get_installed_custom_nodes()
            return web.json_response(custom_nodes)

        @routes.post("/comfy-pilot/chat")
        async def chat(request: web.Request) -> web.StreamResponse:
            """Chat with an agent (streaming response)."""
            data = await request.json()

            agent_name = data.get("agent", "ollama")
            message = data.get("message", "")
            history = data.get("history", [])
            session_id = data.get("session_id", "default")
            current_workflow = data.get("current_workflow")

            # Get the agent backend
            agent = AgentRegistry.get(agent_name)
            if not agent:
                return web.json_response(
                    {"error": f"Agent '{agent_name}' not found"},
                    status=404
                )

            # Check availability
            if not await agent.is_available():
                return web.json_response(
                    {"error": f"Agent '{agent_name}' is not available"},
                    status=503
                )

            # Build messages list
            messages = []

            # Add history
            for msg in history:
                messages.append(AgentMessage(
                    role=msg.get("role", "user"),
                    content=msg.get("content", "")
                ))

            # Add current message
            messages.append(AgentMessage(role="user", content=message))

            # Get system context for the agent
            system_context = await self._build_system_context()

            # Include current workflow if provided
            workflow_context = ""
            if current_workflow:
                workflow_context = self._build_workflow_context(current_workflow)

            config = AgentConfig(
                system_prompt=agent.get_default_system_prompt() + "\n\n" + system_context + "\n\n" + workflow_context
            )

            # Create streaming response
            response = web.StreamResponse(
                status=200,
                reason="OK",
                headers={
                    "Content-Type": "text/plain; charset=utf-8",
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                }
            )
            await response.prepare(request)

            try:
                async for chunk in agent.query(messages, config):
                    await response.write(chunk.encode("utf-8"))
            except Exception as e:
                await response.write(f"\n\nError: {str(e)}".encode("utf-8"))

            await response.write_eof()
            return response

        @routes.post("/comfy-pilot/apply-workflow")
        async def apply_workflow(request: web.Request) -> web.Response:
            """Validate and prepare a workflow for application."""
            data = await request.json()
            workflow = data.get("workflow", {})

            # Validate the workflow
            manipulator = WorkflowManipulator(workflow)
            is_valid, errors = manipulator.validate()

            if not is_valid:
                return web.json_response({
                    "success": False,
                    "errors": errors
                }, status=400)

            return web.json_response({
                "success": True,
                "workflow": workflow,
                "node_count": len(workflow)
            })

    async def _build_system_context(self) -> str:
        """Build system context string for agents."""
        lines = ["## CURRENT SYSTEM STATUS"]

        # GPU info
        gpu_info = await SystemMonitor.get_gpu_info()
        if gpu_info.get("available") and gpu_info.get("gpus"):
            gpu = gpu_info["gpus"][0]
            lines.append(
                f"**GPU**: {gpu['name']}, "
                f"{gpu['vram_free_mb']}MB VRAM free of {gpu['vram_total_mb']}MB"
            )
            # Add VRAM-based recommendations
            vram_free = gpu['vram_free_mb']
            if vram_free < 6000:
                lines.append("  → Low VRAM: Recommend SD 1.5, fp8 models, tiled VAE")
            elif vram_free < 10000:
                lines.append("  → Medium VRAM: SDXL OK, video with fewer frames")
            elif vram_free < 16000:
                lines.append("  → Good VRAM: FLUX fp8 OK, most video workflows")
            else:
                lines.append("  → High VRAM: All models supported")
        else:
            lines.append("**GPU**: Information unavailable")

        # Available models (try to get them)
        try:
            models = await SystemMonitor.get_available_models()
            if models.get("checkpoints"):
                lines.append(f"\n**Available checkpoints**: {', '.join(models['checkpoints'][:5])}")
                if len(models['checkpoints']) > 5:
                    lines.append(f"  ... and {len(models['checkpoints']) - 5} more")
            if models.get("loras"):
                lines.append(f"**LoRAs**: {len(models['loras'])} available")
            if models.get("controlnets"):
                lines.append(f"**ControlNets**: {', '.join(models['controlnets'][:3])}")
        except Exception:
            pass

        # Custom nodes info
        try:
            custom_nodes = await SystemMonitor.get_installed_custom_nodes()
            if custom_nodes.get("found"):
                lines.append(f"\n**Custom nodes installed**: {custom_nodes['total_count']} packs")

                capabilities = custom_nodes.get("node_capabilities", {})
                if capabilities.get("video"):
                    lines.append(f"  - Video: {', '.join(capabilities['video'])}")
                if capabilities.get("face"):
                    lines.append(f"  - Face processing: {', '.join(capabilities['face'])}")
                if capabilities.get("upscale"):
                    lines.append(f"  - Upscaling: {', '.join(capabilities['upscale'])}")
                if capabilities.get("controlnet"):
                    lines.append(f"  - ControlNet: {', '.join(capabilities['controlnet'])}")

                # Note what's missing for common tasks
                missing = []
                if not capabilities.get("video"):
                    missing.append("video generation (AnimateDiff/WAN)")
                if not capabilities.get("face"):
                    missing.append("face processing (Impact-Pack)")
                if not capabilities.get("controlnet"):
                    missing.append("ControlNet preprocessors")

                if missing:
                    lines.append(f"\n  **Missing for full capability**: {', '.join(missing)}")
                    lines.append("  → Suggest installation if user needs these features")
        except Exception:
            pass

        return "\n".join(lines)

    def _build_workflow_context(self, workflow: Dict[str, Any]) -> str:
        """Build context string from the current workflow."""
        lines = ["## CURRENT WORKFLOW (User's active workflow in ComfyUI)"]
        lines.append("The user has shared their current workflow. Analyze it to provide accurate modifications.")
        lines.append("")

        # Extract key information from the workflow
        nodes = workflow.get("nodes", [])
        links = workflow.get("links", [])

        if not nodes:
            lines.append("(Empty workflow)")
            return "\n".join(lines)

        lines.append(f"**Node count**: {len(nodes)}")
        lines.append(f"**Connection count**: {len(links) if links else 0}")
        lines.append("")

        # Group nodes by type
        node_types = {}
        for node in nodes:
            node_type = node.get("type", "Unknown")
            if node_type not in node_types:
                node_types[node_type] = []
            node_types[node_type].append(node)

        lines.append("**Nodes by type**:")
        for node_type, type_nodes in sorted(node_types.items()):
            lines.append(f"- {node_type}: {len(type_nodes)}")

        lines.append("")
        lines.append("**Node details**:")

        # Extract key parameters from important nodes
        for node in nodes:
            node_type = node.get("type", "Unknown")
            node_id = node.get("id", "?")
            title = node.get("title", node_type)

            # Focus on nodes with widgets (parameters)
            widgets = node.get("widgets_values", [])
            if widgets:
                lines.append(f"\n[{node_id}] {title} ({node_type}):")

                # Try to identify common parameter patterns
                if "KSampler" in node_type:
                    self._extract_ksampler_params(lines, widgets, node)
                elif "EmptyLatentImage" in node_type:
                    self._extract_latent_params(lines, widgets)
                elif "CLIPTextEncode" in node_type or "CLIP" in node_type:
                    self._extract_clip_params(lines, widgets)
                elif "VAE" in node_type:
                    self._extract_vae_params(lines, widgets, node_type)
                elif "CheckpointLoader" in node_type:
                    self._extract_checkpoint_params(lines, widgets)
                elif "LoraLoader" in node_type:
                    self._extract_lora_params(lines, widgets)
                elif "ControlNet" in node_type:
                    self._extract_controlnet_params(lines, widgets)
                elif "Video" in node_type or "AnimateDiff" in node_type:
                    self._extract_video_params(lines, widgets, node_type)
                else:
                    # Generic widget display
                    if len(widgets) <= 5:
                        lines.append(f"  widgets: {widgets}")

        lines.append("")
        lines.append("When suggesting modifications, reference specific node IDs and parameter names.")
        lines.append("Provide the exact values to change (from → to).")

        return "\n".join(lines)

    def _extract_ksampler_params(self, lines: List[str], widgets: List, node: Dict) -> None:
        """Extract KSampler parameters."""
        # KSampler typically has: seed, steps, cfg, sampler_name, scheduler, denoise
        param_names = ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"]
        for i, name in enumerate(param_names):
            if i < len(widgets):
                lines.append(f"  {name}: {widgets[i]}")

    def _extract_latent_params(self, lines: List[str], widgets: List) -> None:
        """Extract EmptyLatentImage parameters."""
        param_names = ["width", "height", "batch_size"]
        for i, name in enumerate(param_names):
            if i < len(widgets):
                lines.append(f"  {name}: {widgets[i]}")

    def _extract_clip_params(self, lines: List[str], widgets: List) -> None:
        """Extract CLIP/prompt parameters."""
        if widgets:
            text = str(widgets[0])
            if len(text) > 200:
                text = text[:200] + "..."
            lines.append(f"  prompt: \"{text}\"")

    def _extract_vae_params(self, lines: List[str], widgets: List, node_type: str) -> None:
        """Extract VAE parameters."""
        if "Tiled" in node_type and widgets:
            lines.append(f"  tile_size: {widgets[0] if widgets else 'default'}")

    def _extract_checkpoint_params(self, lines: List[str], widgets: List) -> None:
        """Extract checkpoint loader parameters."""
        if widgets:
            lines.append(f"  checkpoint: {widgets[0]}")

    def _extract_lora_params(self, lines: List[str], widgets: List) -> None:
        """Extract LoRA parameters."""
        param_names = ["lora_name", "strength_model", "strength_clip"]
        for i, name in enumerate(param_names):
            if i < len(widgets):
                lines.append(f"  {name}: {widgets[i]}")

    def _extract_controlnet_params(self, lines: List[str], widgets: List) -> None:
        """Extract ControlNet parameters."""
        param_names = ["strength", "start_percent", "end_percent"]
        for i, name in enumerate(param_names):
            if i < len(widgets):
                lines.append(f"  {name}: {widgets[i]}")

    def _extract_video_params(self, lines: List[str], widgets: List, node_type: str) -> None:
        """Extract video generation parameters."""
        if "AnimateDiff" in node_type:
            lines.append(f"  (AnimateDiff node with {len(widgets)} parameters)")
        elif "Video" in node_type:
            # Try to show frame-related params
            for i, val in enumerate(widgets[:5]):
                lines.append(f"  param_{i}: {val}")


# Global controller instance
controller = ComfyPilotController()
