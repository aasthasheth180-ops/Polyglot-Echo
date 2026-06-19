FROM nginx:alpine

# Copy all your HTML, CSS, and JavaScript files directly into Nginx's web directory
COPY . /usr/share/nginx/html

# Expose Port 80 (the standard port Nginx uses to serve websites)
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]