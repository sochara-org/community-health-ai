import requests
import os


def split_large_para_optimally(para, size):
    s = para
    parts = []
    while (len(s) > size):
        sentence_end = s.rfind('.', 0, size)
        if sentence_end > -1:
            parts.append(s[0:sentence_end + 1])
            s = s[sentence_end + 1:]
        else:
            word_end = s.rfind(' ', 0, size)
            if word_end > -1:
                parts.append(s[0:word_end])
                s = s[word_end + 1:]
            else:
                parts.append(s[0:size])
                s = s[size:]
    if len(s) > 0:
        parts.append(s)
    return [part for part in parts if part != ""]
    


def convert_to_chunks(paragraphs):
    chunks = []
    max_chunk_size = 300
    chunk = ""
    for para in paragraphs:
        if (len(chunk) + len(para)) > max_chunk_size:
            chunks.append(chunk)
            chunk = para
            continue
        parts = split_large_para_optimally(para, max_chunk_size)
        chunks.extend(parts[0:-1])
        chunk += " " + parts[-1]
    if chunk != "":
        chunks.append(chunk)
    return chunks



def get_paragraphs(text):
  """Splits a large body of text into paragraphs based on two or more consecutive newlines.

  Args:
    text: The input text to be split.

  Returns:
    A list of paragraphs.
  """

  paragraphs = []
  current_paragraph = ""

  for line in text.splitlines():
    if line == "":
      # Paragraph end detected
      paragraphs.append(current_paragraph.strip())
      current_paragraph = ""
    else:
      current_paragraph += line.replace("\n", " ") + " "

  if current_paragraph:
    paragraphs.append(current_paragraph.strip())

  return [x for x in paragraphs if x != '']

def extract_pre_text_from_path(html_path):
  try:
    with open(html_path, 'r') as f:
      html_content = f.read()

    return extract_pre_text(html_content)

  except FileNotFoundError:
    print("File not found:", html_path)
    return None


def extract_pre_text(html):
    start_index = html.find('<pre>')
    end_index = html.find('</pre>')

    if start_index != -1 and end_index != -1 and start_index < end_index:
      pre_text = html[start_index + len('<pre>'):end_index]
      return pre_text.strip()
    else:
      return html.strip()


def read_or_download_file(path, url):
  """Reads the content of a file if it exists. If not, downloads it from the specified URL and saves it to the path.

  Args:
    path: The path to the file.
    url: The URL of the file to download.

  Returns:
    The content of the file as a string.
  """

  if os.path.exists(path):
    with open(path, 'r') as f:
      content = f.read()
    return content
  else:
    response = requests.get(url)
    response.raise_for_status()

    with open(path, 'wb') as f:
      f.write(response.content)

    return response.text

def test_split():
    test_file = "sochara.bettercareinlepr0000drvv_djvu.txt"
    content = extract_pre_text(test_file)
    paragraphs = get_paragraphs(content)
    chunks = convert_to_chunks(paragraphs)
    print("Number of chunks", len(chunks))
    print(chunks[10])

def test_large_para_split():
    s = "Testing. Hello"
    print(split_large_para_optimally(s, 5))

if __name__ == "__main__":
    # test_large_para_split()
    test_split()  