from gradio_client import Client
for space_id in ["Lightricks/ltx-video-distilled", "Lightricks/ltx-2-distilled"]:
    print(f"\n{space_id}")
    try:
        print(Client(space_id).view_api())
    except Exception as e:
        print(f"FAILED: {e}")