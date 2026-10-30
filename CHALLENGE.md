# Paired challenge set

Exactly five matched pairs for my solo project. All questions are in the three core languages (Arabic: 2 pairs, Korean: 2, Telugu: 1). These are controlled examples, not a representative test distribution. English contexts are concise paraphrases of public primary sources, not the entire source articles. Answerability is judged only from the supplied context.

Models and thresholds must be frozen before inference. The extension policy was frozen before this set was written. No example may be changed in response to a model output.

## 01-apollo (ar)

- Answerable: متى انطلقت مهمة Apollo 11؟
- English: When did the Apollo 11 mission launch?
- Unanswerable: متى انتهت مهمة Apollo 11؟
- English: When did the Apollo 11 mission end?
- Alteration: event predicate
- Gold span: July 16, 1969
- Why the altered question is unanswerable: The passage dates the launch, not the end of the mission. Apollo 11 and the available date remain plausible distractions.
- [Public source](https://www.nasa.gov/mission/apollo-11/)

Context: On July 16, 1969, NASA sent Apollo 11 into space from Kennedy Space Center. The crew consisted of Neil Armstrong, Buzz Aldrin and Michael Collins. The mission was intended to put people on the Moon.

## 02-mercury (ar)

- Answerable: ما الكوكب الأقرب إلى الشمس؟
- English: Which planet is closest to the Sun?
- Unanswerable: ما الكوكب الأبعد عن الشمس؟
- English: Which planet is farthest from the Sun?
- Alteration: comparative reversal
- Gold span: Mercury
- Why the altered question is unanswerable: The nearest planet is given. The identity of the farthest planet is absent, even though Mercury is a planet name of the expected answer type.
- [Public source](https://science.nasa.gov/mercury/facts/)

Context: Among the planets in our solar system, Mercury has the smallest size and orbits closest to the Sun. Mercury is only a little larger than the Moon that orbits Earth.

## 03-nobel-first (ko)

- Answerable: 첫 번째 노벨 물리학상 수상자는 누구인가요?
- English: Who received the first Nobel Prize in Physics?
- Unanswerable: 두 번째 노벨 물리학상 수상자는 누구인가요?
- English: Who received the second Nobel Prize in Physics?
- Alteration: ordinal change
- Gold span: Wilhelm Conrad Röntgen
- Why the altered question is unanswerable: Only the inaugural award and recipient are described. A fluent second-award question has the same topic and answer type but no supported recipient.
- [Public source](https://www.nobelprize.org/prizes/physics/1901/summary/)

Context: In 1901, the inaugural Nobel Prize in Physics went to Wilhelm Conrad Röntgen. The award recognised his discovery of X-rays.

## 04-nobel-field (ko)

- Answerable: 1911년 노벨 화학상 수상자는 누구인가요?
- English: Who received the 1911 Nobel Prize in Chemistry?
- Unanswerable: 1911년 노벨 의학상 수상자는 누구인가요?
- English: Who received the 1911 Nobel Prize in Medicine?
- Alteration: award-field change
- Gold span: Marie Curie
- Why the altered question is unanswerable: The year and Nobel award remain, but chemistry changes to medicine. The passage says nothing about a medicine laureate.
- [Public source](https://www.nobelprize.org/prizes/chemistry/1911/summary/)

Context: Marie Curie received the chemistry Nobel in 1911 for her investigations of radium and polonium. The prize recognised her contributions to understanding these elements.

## 05-hubble (te)

- Answerable: హబుల్ టెలిస్కోప్‌ను మొదటిసారి ఏ సంవత్సరంలో ప్రయోగించారు?
- English: In which year was the Hubble telescope first launched?
- Unanswerable: హబుల్ టెలిస్కోప్‌ను మొదటిసారి ఏ సంవత్సరంలో మరమ్మతు చేశారు?
- English: In which year was the Hubble telescope first repaired?
- Alteration: event predicate
- Gold span: 1990
- Why the altered question is unanswerable: The passage gives the launch year but no repair year. The same telescope and a plausible-looking year remain in the context.
- [Public source](https://science.nasa.gov/mission/hubble/overview/about-hubble/)

Context: The space shuttle Discovery carried the Hubble Space Telescope into space in 1990. This observatory takes its name from astronomer Edwin Hubble.

