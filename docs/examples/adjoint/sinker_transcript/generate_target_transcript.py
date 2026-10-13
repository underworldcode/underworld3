"""Twin-experiment target for the transcript forward model: run at the true centre,
save v(T). Kept on disk (not in memory) so a candidate run on a freshly built
mesh reads it back through the coordinate-remapping `read_timestep`, exactly as
the original project does."""
import os
import underworld3 as uw
from forward_sinker_transcript import build_model, solve_forward, BLOB_CENTER

TRUE_VISCOSITY_CONTRAST = 1000.0
TARGET_FILENAME = "sinker_target_transcript"
TARGET_INDEX = 0
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "target")

if __name__ == "__main__":
    if uw.mpi.rank == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = build_model(TRUE_VISCOSITY_CONTRAST)
    transcript, _ = solve_forward(model, BLOB_CENTER)
    model["mesh"].write_timestep(TARGET_FILENAME, index=TARGET_INDEX,
                                 outputPath=OUTPUT_DIR, meshVars=[model["v"]])
    uw.pprint(f"true centre {BLOB_CENTER}; {len(transcript)} steps; "
              f"saved v(T) to {OUTPUT_DIR}/{TARGET_FILENAME}")
