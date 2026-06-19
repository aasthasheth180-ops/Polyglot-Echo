FROM nginx:alpine

# Completely clear out Nginx's default welcome pages
RUN rm -rf /usr/share/nginx/html/*

# Copy your frontend files into the Nginx container directory
COPY . /usr/share/nginx/html

# Tell Nginx to look for stream_test.html instead of index.html
RUN sed -i 's/index.html/stream_test.html/g' /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]