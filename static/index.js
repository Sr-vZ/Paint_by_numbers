$(function () {
    $('#afterImage').hide()
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();

            reader.onloadend = function (e) {
                $('#imgThumb').attr('src', e.target.result);
                $('#afterImage').show()
            }

            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#customFile").change(function () {
        readURL(this);
    })

    $('#customRange2Value').html($('#customRange2').val())
    $('#customRange2').on('change',function () {
        console.log(this.value)
        $('#customRange2Value').html(this.value)
    })
    $("#submitBtn").click(function () {
        $.ajax({
            type: "GET",
            url: "/success",
            // data: dataString,
            success: function (response, status) {
                console.log(response)
                $('#imgThumb').attr('src', response.filename);
            }
        })
    })
    
})