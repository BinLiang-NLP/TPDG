The files in this folder provide three types of manual annotations: stance, target of opinion, and sentiment. The additional annotations (target of opinion and sentiment) were not part of the SemEval-2016 competiton, but are made available for future research. Details about this dataset are available in the following paper:

Saif M. Mohammad, Parinaz Sobhani, and Svetlana Kiritchenko. 2016. Stance and sentiment in tweets. Special Section of the ACM Transactions on Internet Technology on Argumentation in Social Media, Submitted. 


************************************************
DATA ANNOTATION
************************************************

The data annotations files ("trialdata-all-annotations.txt", "trainingdata-all-annotations.txt", "testdata-taskA-all-annotations.txt", and "testdata-taskB-all-annotations.txt") have the following format:
<ID><tab><Target><tab><Tweet><tab><Stance><tab><Opinion towards><tab><Sentiment>

where
<ID> is an internal identification number;
<Target> is the entity of interest (e.g., "Hillary Clinton");
<Tweet> is the text of a tweet;
<Stance> is the stance label;
<Opinion towards> is the label for the target of opinion;
<Sentiment> is the sentiment label.


The possible stance labels are:
1. FAVOR: We can infer from the tweet that the tweeter supports the target (e.g., directly or indirectly by supporting someone/something, by opposing or criticizing someone/something opposed to the target, or by echoing the stance of somebody else).
2. AGAINST: We can infer from the tweet that the tweeter is against the target (e.g., directly or indirectly by opposing or criticizing someone/something, by supporting someone/something opposed to the target, or by echoing the stance of somebody else).
3. NONE: none of the above.


The possible 'opinion towards' labels are:
1. TARGET: The tweet explicitly expresses opinion about the target, a part of the target, or an aspect of the target.
2. OTHER: The tweet does NOT expresses opinion about the target but it HAS opinion about something or someone other than the target.
3. NO ONE: The tweet is not explicitly expressing opinion. (For example, the tweet is simply giving information.)


The possible sentiment labels are:
1. POSITIVE: the speaker is using positive language, for example, expressions of support, admiration, positive attitude, forgiveness, fostering, success, positive emotional state (happiness, optimism, pride, etc.)
2. NEGATIVE: the speaker is using negative language, for example, expressions of criticism, judgment, negative attitude, questioning validity/competence, failure, negative emotional state (anger, frustration, sadness, anxiety, etc.)
3. NEITHER: none of the above.



************************************************
Tweet IDs
************************************************

The ids files ("trialdata-ids.txt", "trainingdata-ids.txt", "testdata-taskA-ids.txt", and "testdata-taskB-ids.txt") have the following format:
<ID><tab><Tweet ID>

where
<ID> is an internal identification number (same as used in the data annotation files);
<Tweet ID> is a tweet ID for the original tweet. Please note that the tweets in the trial, training and test data are not exact copies of the original tweets posted on Twitter. For the details, please refer to the task description paper:

Mohammad, S., Kiritchenko, S., Sobhani, P., Zhu, X., and Cherry, C. SemEval-2016 Task 6: Detecting Stance in Tweets. Proceedings of the International Workshop on Semantic Evaluation (SemEval-2016), San Diego, California, 2016.



