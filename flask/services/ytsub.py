import pytube as pt
import subprocess
import webvtt
import nltk
from nltk.tokenize import word_tokenize
import os

def main(link):
    yt = pt.YouTube(link)
    print(yt.title)
    # import vttToTxt as convert
    # Command to be executed
    command = f"sudo yt-dlp --verbose --write-auto-sub --sub-lang en --skip-download --id {link}"

    # Run the command and capture the output
    try:
        # Run the command in a subprocess, capture output and errors
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check if the command was successful
        if result.returncode == 0:
            # Print the output of the command
            print("Command output:")
            print(result.stdout)
        else:
            # Print the error message if the command failed
            print("Error occurred:")
            print(result.stderr)

    except Exception as e:
        # Handle exceptions, if any
        print("Exception occurred:", str(e))

    # filename=yt.title+" ["+yt.video_id+"].en"
    filename=yt.video_id+".en"
    print(filename)
    print(1)
    import time
    time.sleep(5)
    vtt=[]
    try:
        vtt = webvtt.read(filename+".vtt")
        
        print(1)
    except Exception as e:
        print(e)
    transcript=""

    lines = []
    for line in vtt:
        # Strip the newlines from the end of the text.
        # Split the string if it has a newline in the middle
        # Add the lines to an array
        lines.extend(line.text.strip().splitlines())
    print(1)

    # Remove repeated lines
    previous = None
    
    for line in lines:
        if line == previous:
            continue
        transcript += " " + line
        previous = line
    print(1)

    # print(transcript)
    print("transcript generated")




    # Download the NLTK tokenizer data (if not already downloaded)
    
    nltk.download('punkt')

    # Function to split text into chunks of approximately 3000 tokens per file
    def split_text_into_chunks(input_text, chunk_size=500):
        tokens = word_tokenize(input_text)
        num_chunks = len(tokens) // chunk_size + (len(tokens) % chunk_size != 0)
        chunks = [tokens[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
        return chunks

    # Function to write chunks to separate text files
    # def write_chunks_to_files(chunks, output_directory):
    #     os.makedirs(output_directory, exist_ok=True)
    #     for i, chunk in enumerate(chunks):
    #         with open(os.path.join(output_directory, f'chunk_{i + 1}.txt'), 'w', encoding='utf-8') as file:
    #             file.write(' '.join(chunk))

    def write_chunks_to_files(chunks, output_directory):
        os.makedirs(output_directory, exist_ok=True)
        for i, chunk in enumerate(chunks):
            file_path = os.path.join(output_directory, f'chunk_{i + 1}.txt')
            # Use subprocess to write to files
            with subprocess.Popen(['sudo','tee', file_path], stdin=subprocess.PIPE, universal_newlines=True) as process:
                process.communicate(input=' '.join(chunk))




    try:
        # Example usage
    # input_text = "Your input text goes here. This could be a long piece of text that you want to split into chunks."
        output_directory = os.getcwd()+"/data/output_chunks"
        print(output_directory)
        # print(f"transcript: {transcript}")
        chunks = split_text_into_chunks(transcript)
        write_chunks_to_files(chunks, output_directory)
    except Exception as e:
        print(e)
    
    # Split text into chunks and write to separate files
    

    # convert.main(filename) old way
# main("https://www.youtube.com/watch?v=KkUSb4y_sV0")