from textSummarizer.pipeline.prediction import PredictionPipeline

text = """Yeah I've seen this pattern a lot with DSA prep.

Most people think they're "not getting it" when really they're just doing random problem sets without any feedback loop or structure. You end up revisiting the same gaps over and over.

That said, tools like this can help a bit with direction, but they don't replace the actual grind of understanding patterns. At some point you still have to sit with problems long enough for them to click.

If it actually tracks weaknesses properly and doesn't just recycle difficulty-based recommendations, it could be useful for beginners who don't know what to focus on next. But that's a big "if" with a lot of these platforms.

Ngl though, even simple stuff like maintaining your own mistake log in Notion can get you 70% of the benefit without needing another tool."""

try:
    obj = PredictionPipeline()
    print("Prediction Pipeline initialized successfully.\n")
    summary = obj.predict(text)
    print("\n--- Final Summary ---")
    print(summary)
except Exception as e:
    import traceback
    traceback.print_exc()
