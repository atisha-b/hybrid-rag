import config
client = config.qdrant_client()
hits = client.scroll(
    collection_name=config.FIGURES_COLLECTION,
    limit=50,
    with_payload=True,
)[0]
for h in hits:
    p = h.payload
    print(f"page={str(p.get('page','?')):>4}  caption={str(p.get('caption',''))[:90]}")
