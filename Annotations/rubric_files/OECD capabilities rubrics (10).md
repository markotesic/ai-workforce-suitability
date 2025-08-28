# **Perception and Pattern Recognition**

This rubric measures how much a task depends on detecting and interpreting salient details, e.g. signals, regularities, anomalies, or subtle cues, by drawing on prior knowledge. At the easiest levels, surface patterns are obvious and almost everyone spots them. At the hardest, success demands synthesising faint, multi-layered regularities buried in noisy textual data or narratives that only a tiny fraction of people can reliably discern.

## **Level 0: None**

The task needs no pattern recognition; copying or basic recall suffices.  Performance is unaffected by ignoring all contextual or relational cues.

**Examples**:

* Copy a provided numerical string exactly.  
* Convert “hello” to uppercase.  
* Select answer ‘C’ because the prompt explicitly says “Choose C”.

## **Level 1: Very low**

Requires noticing a single, highly salient cue or simple pattern that is obvious to most people. Recognition is nearly automatic for most readers.

**Examples**:

* In “The meeting is on Friday at 10 am,” extract the day of the week  
* Identify which of the following three lines of code contains the repeated trigram ‘abc’.  
* Given the sequence ‘*A B A B A B’*, supply the next two letters.

## **Level 2: Low**

The solver must recognise a simple but non-obvious pattern, for instance, a basic trend or a two-feature correlation, across a short span of data. A small amount of prior knowledge or inference improves performance, but the rule remains concrete and explicit once seen.

**Examples**:

* From a six-row sales table, identify the first month where revenue begins a consistent decline.  
* In a list of ten sentences, flag each one that contains a prime number written in words (e.g., “seven,” “eleven”).  
* Given ten timestamped log entries, mark the three that break an every-15-minutes rhythm.

## **Level 3: Medium**

Tasks require integrating multiple cues or cross-referenced details, often with distractors, to spot an anomaly, hidden rule, or foreshadowing device.  The pattern is discoverable but rarely obvious.

**Examples:**

* Examine ten HTTP logs and identify the single entry that signals a breach by combining an unusual status code with an out-of-range timestamp.  
* In a 500-word short story, highlight the sentence that foreshadows the twist ending via a double meaning.  
* Given eight brief medical case notes, spot the one patient whose lab pattern contradicts the cluster description.

## **Level 4: High**

The solver must uncover layered or shifting patterns within noisy, partially relevant data, often merging domain knowledge with logical elimination. Cues may conflict or be sparsely distributed, and several plausible distractions exist.

**Examples:**

* Analyse a 40-line Slack thread to deduce which employee leaked data, using subtle time-zone hints and vocabulary idiosyncrasies.  
* From a summary of 30 experiments, identify a hidden confounder that causes all trials with a particular protocol to fail only under humidity \> 80%.  
* Parse a draft policy full of cross-references and pinpoint the clause that unintentionally overrides two earlier guarantees.

## **Level 5: Very high**

Success demands detecting faint, multi-layer regularities or covert structures across lengthy, noisy data where patterns nest, shift or are adversarially obfuscated.

**Examples**:

* From a 10000-word email archive, trace an insider-trading scheme by linking coded phrases, irregular send times, and repeating checksum strings.  
* Analyse a 50-page detective story told in shuffled diary snippets to identify the true culprit, using mirrored dates and inconsistent pronouns.  
* Review a 300-page fantasy novel manuscript and uncover an acrostic prophecy hidden in the first letters of each chapter title, noting how the message is distorted by deliberate misspellings and chapter re-ordering.

