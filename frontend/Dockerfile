FROM nginx:alpine

# Completely clear out Nginx's default welcome pages
RUN rm -rf /usr/share/nginx/html/*

# Fix: Only copy the contents of the 'frontend' folder
COPY frontend/ /usr/share/nginx/html/

# Tell Nginx to look for stream_test.html
RUN sed -i 's/index.html/index.html/g' /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]