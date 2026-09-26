from vision_base.apps import ConfigApp
from vision_base.hooks import Outgoing

class DoorCounter(ConfigApp):              # analyzers still from config
    name = "door-counter"
    def on_event(self, ctx, ev):
        out = super().on_event(ctx, ev)    # keep default events/<stream_id>
        if ev.type == "line_cross":
            n = ctx.state.get("inside", 0) + (1 if ev.fields["direction"] == "forward" else -1)
            ctx.state["inside"] = n
            out.append(Outgoing(f"count/{ctx.stream_id}", {"inside": n}))
        return out
