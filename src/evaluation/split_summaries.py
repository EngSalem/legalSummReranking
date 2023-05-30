import pandas as pd

df = pd.read_csv('../../src/summarization/outputs/test_irc_tags_with_irc_content.csv')

df['generated_summary'] = df.apply(lambda row: row['generated_summary'].split('[SUMMARY]')[-1], axis=1)
df.to_csv('../../src/summarization/outputs/test_irc_tags_with_irc_content_summary_only.csv', index=False)