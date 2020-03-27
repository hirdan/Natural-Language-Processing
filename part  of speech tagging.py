import nltk
import numpy
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer  #unsuperised learning tokenizer

## pos tag list
##Number  Tag     Description
##1.	CC	Coordinating conjunction
##2.	CD	Cardinal number
##3.	DT	Determiner
##4.	EX	Existential there
##5.	FW	Foreign word
##6.	IN	Preposition or subordinating conjunction
##7.	JJ	Adjective
##8.	JJR	Adjective, comparative
##9.	JJS	Adjective, superlative
##10.	LS	List item marker
##11.	MD	Modal
##12.	NN	Noun, singular or mass
##13.	NNS	Noun, plural
##14.	NNP	Proper noun, singular
##15.	NNPS	Proper noun, plural
##16.	PDT	Predeterminer
##17.	POS	Possessive ending
##18.	PRP	Personal pronoun
##19.	PRP$	Possessive pronoun
##20.	RB	Adverb
##21.	RBR	Adverb, comparative
##22.	RBS	Adverb, superlative
##23.	RP	Particle
##24.	SYM	Symbol
##25.	TO	to
##26.	UH	Interjection
##27.	VB	Verb, base form
##28.	VBD	Verb, past tense
##29.	VBG	Verb, gerund or present participle
##30.	VBN	Verb, past participle
##31.	VBP	Verb, non-3rd person singular present
##32.	VBZ	Verb, 3rd person singular present
##33.	WDT	Wh-determiner
##34.	WP	Wh-pronoun
##35.	WP$	Possessive wh-pronoun
##36.	WRB	Wh-adverb
##

## Named Entity Type Examples
##ORGANIZATION      Georgia_Pacific Corp., WHO
##PERSON       Eddy Bonte, President Obama
##LOCATION    Murray River, Mount Everest
##DATE   June, 2008-06-29
##TIME    two fifty a m, 1:30 p.m.
##MONEY    175 million Canadian Dollars, GBP 10.40
##PERCENT   twenty pct, 18.75 %
##FACILITY    Washington Monument, Stonehenge
##GPE     South East Asia, Midlothian
##

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

#chunking

##            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
##
##            chunkParser = nltk.RegexpParser(chunkGram)
##            chunked = chunkParser.parse(tagged)

            #print(chunked)
            #print(tagged)
            #chunked.draw()

#chinking            
##            chunkGram = r"""Chunk: {<.*>+}
##                                    }<VB.?|IN|DT>+{"""
##            chunkParser = nltk.RegexpParser(chunkGram)
##            chunked = chunkParser.parse(tagged)
##            chunked.draw()

#named entity recognition

            namedEnt = nltk.ne_chunk(tagged, binary=True)
            namedEnt.draw()

    except Exception as e:
        print(str(e))

process_content()


