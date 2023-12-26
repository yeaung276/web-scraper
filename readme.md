problem 1: google search on the site and serper scraper has difference. see what parameter do I need to tweake to get best result. Search result mentioned same site multiple time.

problem 2: scraping on agent protected sites.

problem 3: extracting exact text for description. This prompt promising.
collect exact text content that is description of product "Pax Plus".
description

problem 4: if you want all description, gpt api gonna take a long response time.
this prompt get most of the description but take too long
            Remove all text content that is not part of the description of the product "{query}".
            description: {text}
this prompt get only some of the description but is a lot quicker than other
            Collect all exact text content that describe of product "{query}" from raw text.
            raw text: {text}
            description:

problem 5: map reduce function seem to not doing gpt request concurrently

problem 6: because of async scraping, the streaming order is not maintained, finish first will be sent first to the client

v2 is based on Q&A retreieveal process

learn something to improve processing time:
https://medium.com/@entrustech/perplexity-ai-what-you-need-to-know-and-how-to-use-it-82ee6ce1fbd#:~:text=to%20user%20queries.-,It%20is%20designed%20to%20search%20the%20web%20in%20real%2Dtime,and%20generate%20human%2Dlike%20text.

profile the request/response
https://medium.com/@vpcarlos97/unleash-the-power-of-profiling-python-api-requests-effectively-with-profyle-fecac800ed66#:~:text=Profyle%20is%20a%20Python%20package,just%201%20line%20of%20code

