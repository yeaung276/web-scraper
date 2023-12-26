async function readAllChunks(readableStream) {
    const reader = readableStream.getReader();
    const decoder = new TextDecoder("utf-8")
    
    let done, value;
    while (!done) {
      ({ value, done } = await reader.read());
      if (done) {
        return;
      }
      console.log(decoder.decode(value))
    }
  }
  
  console.log(await readAllChunks((await fetch('http://localhost:8000/search-v1?query="Arizer Solo 2"')).body));