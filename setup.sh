mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"vinayakpatil2665@gmail,com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/credentials.toml
