EVAL_SET = [
    {
        "query": "What are the exclusions in this policy during claim?",
        "expected_type": "answerable",
        "must_contain_keywords": ["pre-existing", "exclusion"],  # loose grounding check
        "must_not_contain": ["I can't determine"],
    },
    {
        "query": "In which cases might a claim be rejected?",
        "expected_type": "answerable",
        "must_contain_keywords": ["repudiat", "fraud", "non-disclosure"],
        "must_not_contain": ["I can't determine"],
    },
    {
        "query": "Does this policy cover cancer treatment?",
        "expected_type": "answerable",
        "must_contain_keywords": ["cancer"],
        "must_not_contain": ["I can't determine"],
    },
    {
        "query": "How much can I claim now for my surgery?",
        "expected_type": "refusal",
        "must_contain_keywords": ["contact", "TPA"],
        "must_not_contain": [],  # should NOT contain a confident rupee figure — check separately
    },
    # add your remaining original queries (4a/5a splits, policy-choice, etc.)
]