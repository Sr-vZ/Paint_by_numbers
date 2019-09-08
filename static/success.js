data = []
$(function () {
    $.ajax({
        type: "GET",
        url: "/get_data",
        // data: dataString,
        success: function (response, status) {
            data = response
            console.log(response)
            // $('#imgThumb').attr('src', response.filename);
        }
    })
})