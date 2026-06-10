"""Core generation engine.

The engine is the application. Front-ends are thin. The dependency flow is:

    UI -> BatchGenerationRequest -> BatchPlanner -> PromptBuilder -> Provider -> Storage/Metadata

The central runtime object is :class:`app.core.models.Run`: it owns the
request, the cost estimate, the plan, the generated assets, the failures and
the on-disk metadata for one generation run.
"""
