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
    def write_transcript_to_file(transcript,output_directory):
        with subprocess.Popen(['sudo','tee',output_directory ], stdin=subprocess.PIPE, universal_newlines=True) as process:
            process.communicate(input=transcript)

    try:
        # Example usage
        output_directory = os.getcwd()+"/data/subtitle.txt"
        print(output_directory)
        print(f"transcript: {transcript}")
        write_transcript_to_file(transcript, output_directory)
        print("subtitle file generated")
    except Exception as e:
        print(e)
    
    # Split text into chunks and write to separate files
    # WRITING
    

    # convert.main(filename) old way
# main("https://www.youtube.com/watch?v=KkUSb4y_sV0")